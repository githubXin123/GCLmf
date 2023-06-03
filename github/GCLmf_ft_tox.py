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

from dataset.dataset_ft_tox import MolTestDatasetWrapper

import timm
import timm.optim
import timm.scheduler
from torch.optim import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft_tox.yaml', os.path.join(model_checkpoints_folder, 'config_ft_tox.yaml'))

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
                weights = [1, float(config['ration_pos'])]
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
            for bn, data in enumerate(train_loader):
            
                optimizer.zero_grad()
                data = data.to(self.device, non_blocking=True)
                loss = self._step(model, data)
                loss.backward()

                optimizer.step()
                n_iter += 1

            print('epoch：', epoch_counter)
            if self.config['dataset']['task'] == 'classification':
                valid_loss, valid_cls = self._validate(model, valid_loader)
            else:
                valid_loss, valid_rgr = self._validate(model, valid_loader)
            self._test(model, test_loader, model_num = None)
                
            if epoch_counter >= (self.config['epochs'])/2:
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

# def set_random_seed(seed=2020):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    config = yaml.load(open("config_ft_tox.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'Cardiotoxicity1':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Cardiotoxicity1.csv'
        target_list = ["Cardiotoxicity1"]

    elif config['task_name'] == 'Cardiotoxicity5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Cardiotoxicity5.csv'
        target_list = ["Cardiotoxicity5"]

    elif config['task_name'] == 'Cardiotoxicity10':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Cardiotoxicity10.csv'
        target_list = ["Cardiotoxicity10"]
    
    elif config['task_name'] == 'Cardiotoxicity30':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Cardiotoxicity30.csv'
        target_list = ["Cardiotoxicity30"]

    elif config['task_name'] == 'Carcinogenicity':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Carcinogenicity.csv'
        target_list = ["Carcinogenicity"]
        
    elif config['task_name'] == 'Ames':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Ames.csv'
        target_list = ["Ames_tox"]

    elif config['task_name'] == 'CYP1A2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/CYP1A2.csv'
        target_list = ["CYP1A2"]
        
    elif config['task_name'] == 'CYP2C9':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/CYP2C9.csv'
        target_list = ["CYP2C9"]

    elif config['task_name'] == 'CYP2C19':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/CYP2C19.csv'
        target_list = ["CYP2C19"]

    elif config['task_name'] == 'CYP2D6':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/CYP2D6.csv'
        target_list = ["CYP2D6"]

    elif config['task_name'] == 'CYP3A4':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/CYP3A4.csv'
        target_list = ["CYP3A4"]

    elif config['task_name'] == 'Acute_oral_toxicity':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/tox/Acute_oral_toxicity.csv'
        target_list = ["Acute_oral_toxicity"]
        
    elif config['task_name'] == 'Resp_tox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Resp_tox.csv'
        target_list = ["Resp_tox"]
        
    elif config['task_name'] == 'EC':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/EC.csv'
        target_list = ["EC"]

    elif config['task_name'] == 'EI':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/EI.csv'
        target_list = ["EI"]

    elif config['task_name'] == 'BCF':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/tox/BCF.csv'
        target_list = ["BCF"]
    
    elif config['task_name'] == 'LC50':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/tox/LC50.csv'
        target_list = ["LC50"]
        
    elif config['task_name'] == 'LC50DM':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/tox/LC50DM.csv'
        target_list = ["LC50DM"]

    elif config['task_name'] == 'IGC50':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/tox/IGC50.csv'
        target_list = ["IGC50"]
    
    elif config['task_name'] == 'NR_AhR':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_AhR.csv'
        target_list = ["NR_AhR"]
        
    elif config['task_name'] == 'NR_AR':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_AR.csv'
        target_list = ["NR_AR"]
        
    elif config['task_name'] == 'NR_AR_LBD':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_AR_LBD.csv'
        target_list = ["NR_AR_LBD"]
        
    elif config['task_name'] == 'NR_ER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_ER.csv'
        target_list = ["NR_ER"]
        
    elif config['task_name'] == 'NR_Aromatase':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_Aromatase.csv'
        target_list = ["NR_Aromatase"]
    
    elif config['task_name'] == 'NR_ER_LBD':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_ER_LBD.csv'
        target_list = ["NR_ER_LBD"]
    
    elif config['task_name'] == 'SR_ARE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/SR_ARE.csv'
        target_list = ["SR_ARE"]
        
    elif config['task_name'] == 'SR_ATAD5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/SR_ATAD5.csv'
        target_list = ["SR_ATAD5"]
        
    elif config['task_name'] == 'SR_HSE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/SR_HSE.csv'
        target_list = ["SR_HSE"]
        
    elif config['task_name'] == 'SR_p53':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/SR_p53.csv'
        target_list = ["SR_p53"]

    elif config['task_name'] == 'NR_PPAR_gamma':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/NR_PPAR_gamma.csv'
        target_list = ["NR_PPAR_gamma"]
        
    elif config['task_name'] == 'SR_MMP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/SR_MMP.csv'
        target_list = ["SR_MMP"]
        
    elif config['task_name'] == 'Hepatotoxicity':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox/Hepatotoxicity.csv'
        target_list = ["Hepatotoxicity"]
    
    elif config['task_name'] == 'Urinary_tract_toxicity':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/tox/Urinary_tract_toxicity.csv'
        target_list = ["Urinary_tract_toxicity"]
        
    else:
        raise ValueError('Undefined downstream task!')
        
    print(config)
    
    # set_random_seed(seed=2020)
    
    # for i in range(3):
    for target in target_list:
        print(target)
        config['dataset']['target'] = target
        result = main(config)
