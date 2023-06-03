import os
import shutil
import sys
import yaml
import numpy as np
import random
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, recall_score, matthews_corrcoef, f1_score, precision_score, r2_score

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
        
        
        if config['dataset']['task'] == 'classification':   #针对不同类型的任务选择不同损失函数
            if config['add_weight'] == True:
                print('对类别进行加权')
                weights = [float(config['ration_neg']), float(config['ration_pos'])]      #括号内为两个类别样本的权重：(样本较多的类别的样本数/类别1，样本较多的类别的样本数/类别2)
                class_weights = torch.FloatTensor(weights).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data):
        __, pred = model(data)
        
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        else:
            loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()   

        self.normalizer = None

        from models.ginet_ft import GINet
        model = GINet(self.config['dataset']['task'], pred_n_layer=self.config['pred_n_layer'], **self.config["model"]).to(self.device)
        # print(model)
        model = self._load_pre_trained_weights(model)

        layer_list = []
        
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                # print(name, param.requires_grad)
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
        
        best_valid_rgr = 0
        
        best_valid_cls = 0
        
        for epoch_counter in range(self.config['epochs']):
            loss_train = 0
            for bn, data in enumerate(train_loader):
            
                optimizer.zero_grad()
                data = data.to(self.device, non_blocking=True)
                loss = self._step(model, data)
                loss_train += loss.item()
                loss.backward()
            
                optimizer.step()
                n_iter += 1

            epoch_counter += 1

            print('epoch：', epoch_counter)
            # print('epoch：', epoch_counter, 'train loss:', loss_train/bn)
            if self.config['dataset']['task'] == 'classification':
                valid_loss, valid_cls = self._validate(model, valid_loader)
            else:
                valid_loss, valid_rgr = self._validate(model, valid_loader)
            self._test(model, test_loader, model_num = None)
                
            if epoch_counter >= (self.config['epochs'])/2:
            # if epoch_counter >= 0:
                if self.config['dataset']['task'] == 'classification': 
                    if valid_cls > best_valid_cls:
                        best_valid_cls = valid_cls
                        model_num = epoch_counter
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    if valid_rgr > best_valid_rgr:
                        best_valid_rgr = valid_rgr
                        model_num = epoch_counter
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

        print('the path of model', self.file_name)
        print('the epoch of best performance:', model_num)
            
        self._test(model, test_loader, model_num = model_num)

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
                data = data.to(self.device, non_blocking=True)
                
                __, pred = model(data)
                
                loss = self._step(model, data)
                
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
            R2 = round(r2_score(labels, predictions), 4)
            print('Validation loss:', valid_loss, 'valid_R2:', R2)
            return valid_loss, R2

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = round(roc_auc_score(labels, predictions[:,1]), 4)
            
            print('Validation loss:', valid_loss, 'Valid ROC AUC:', roc_auc)
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
        label_pred = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device, non_blocking=True)

                __, pred = model(data)
                loss = self._step(model, data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

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
            pass
        else:
            if self.config['dataset']['task'] == 'regression':
                predictions = np.array(predictions)
                labels = np.array(labels)
                self.r2 = round(r2_score(labels, predictions), 4)
                print('Test loss:', test_loss, 'Test_R2:', self.r2)
                return self.r2
            elif self.config['dataset']['task'] == 'classification': 
                predictions = np.array(predictions)
                labels = np.array(labels)
                self.roc_auc = round(roc_auc_score(labels, predictions[:,1]), 4)
                label_pred.extend(np.argmax(predictions,axis=1))
                
                print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
                return self.roc_auc

def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        return fine_tune.r2

def set_random_seed(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        
    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/property/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    else:
        raise ValueError('Undefined downstream task!')
        
    print(config)
    
    
    for index, target in enumerate(target_list):
        if index == 0:
            print(target)
            config['dataset']['target'] = target
            result = main(config)
            # results_list.append([target, result])
        else:
            pass

