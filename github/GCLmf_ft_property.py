import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import random
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_ft_molproperty import MolTestDatasetWrapper
import timm
import timm.optim
import timm.scheduler
from torch.optim import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft_property.yaml', os.path.join(model_checkpoints_folder, 'config_ft_property.yaml'))

class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        
        self.file_name = dir_name
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter): 
        # get the prediction
        __, pred = model(data)
                
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()   

        self.normalizer = None

        from models.ginet_ft import GINet
        model = GINet(self.config['dataset']['task'], pred_n_layer=self.config['pred_n_layer'], **self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        layer_list = []

        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                
                layer_list.append(name)
        
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
        
        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
            
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
            
                optimizer.zero_grad()

                data = data.to(self.device, non_blocking=True)
                loss = self._step(model, data, n_iter)
                
                loss.backward()

                optimizer.step() 
                n_iter += 1
            
            print('epoch：', epoch_counter)
            
            if self.config['dataset']['task'] == 'classification':
                valid_loss, valid_cls = self._validate(model, valid_loader)
            else:
                valid_loss, valid_rgr = self._validate(model, valid_loader)
            self._test(model, test_loader, model_num = None)
            
            
            # validate the model if requested
            if epoch_counter >= (self.config['epochs'])/2:
                if self.config['dataset']['task'] == 'classification': 
                    if valid_cls > best_valid_cls:
                        best_valid_cls = valid_cls
                        model_num = epoch_counter
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    if valid_rgr < best_valid_rgr:
                        best_valid_rgr = valid_rgr
                        model_num = epoch_counter
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

        print('the path of model', self.file_name)
        print('best epoch:', model_num)
        self._test(model, test_loader, model_num)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'])
            state_dict = torch.load(os.path.join(checkpoints_folder, self.config['model_num']), map_location=self.device)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad(): 
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())

                else:
                    predictions.extend(pred.cpu().detach().numpy())

                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            print('Validation loss:', valid_loss, 'RMSE:', rmse)
            return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader, model_num):
        if model_num == None:
            pass
        else:
            model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)

        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if model_num == None:
            print()
        else:
            if self.config['dataset']['task'] == 'regression':
                predictions = np.array(predictions)
                labels = np.array(labels)
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

            elif self.config['dataset']['task'] == 'classification': 
                predictions = np.array(predictions)
                labels = np.array(labels)
                self.roc_auc = roc_auc_score(labels, predictions[:,1])
                print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)

        


def main(config): 
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        return fine_tune.rmse

if __name__ == "__main__":
    config = yaml.load(open("config_ft_property.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/property/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/property/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/property/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/property/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/property/freesolv.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/property/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/property/Lipophilicity.csv'
        target_list = ["exp"]
    
    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)
        results_list.append([target, result])

