import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime
from queue import Queue
from threading import Thread

import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent_loss import NTXentLoss
import timm
import timm.optim
import timm.scheduler

from bisect import bisect_right

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_pretrain.yaml', os.path.join(model_checkpoints_folder, 'config_pretrain.yaml'))

class GCLmf(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)   
        self.model_path = log_dir
        # self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.mol_embedding  = ['xj_1','xj_2','xj_3','xj_4','xj_5','xj_6',
                                'xj_7','xj_8','xj_9','xj_10','xj_11','xj_12',
                                'xj_13','xj_14','xj_15','xj_16','xj_17','xj_18',
                                'xj_19','xj_20','xj_21','xj_22','xj_23','xj_24',                  
                                ] 
        
        
    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, b, n_iter):
    
        _, xj_00 = model(b[0])
        for i in range(24):
            _, globals()[self.mol_embedding[i]] = model(b[i+2])
    
        loss_input = []
        xj_00_n = F.normalize(xj_00, dim=1)
        loss_input.append(xj_00_n)
        loss_input.append(xj_00_n)
        
        for i in range(24):
            globals()[self.mol_embedding[i]] = F.normalize(globals()[self.mol_embedding[i]], dim=1)
            loss_input.append(globals()[self.mol_embedding[i]])

        loss = self.nt_xent_criterion(loss_input, n_iter)
        return loss, xj_00, xj_00

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        from models.ginet_pretrain import GINet     
        model = GINet(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )   
            

        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer = optimizer,
            t_initial = self.config['epochs'],
            lr_min = 1e-6,
            warmup_t = self.config['warm_up'], 
            warmup_lr_init = 0.0001
        )
        

        model_checkpoints_folder = os.path.join(self.model_path, 'checkpoints') 
        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        
        for epoch_counter in range(self.config['epochs']):
            scheduler.step(epoch_counter)
            print("LearningRate of %d epoch: %f" % (epoch_counter, optimizer.param_groups[0]['lr']))
            
            for bn, (a) in enumerate(train_loader):
            
                optimizer.zero_grad()
                for i in range(26):
                    a[i] = a[i].to(self.device, non_blocking=True)

                loss, zis, zjs = self._step(model, a, n_iter) 


                if n_iter % self.config['log_every_n_steps'] == 0:  
                    # self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('cosine_lr_decay', optimizer.param_groups[0]['lr'], global_step=n_iter)

                    print('--training--','epoch: ', epoch_counter, 'miniBatch: ', bn, 'loss: ', loss.item()) 

                loss.backward()
                
                optimizer.step()
                n_iter += 1
                
            
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)    
                print('epoch:', epoch_counter, 'valid loss:', valid_loss)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                # self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            #save model
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))
            
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'])
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_4.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():   
            model.eval()
            valid_loss = 0.0
            counter = 0
            for (d) in valid_loader:
                for i in range(26):
                    d[i] = d[i].to(self.device, non_blocking=True)

                loss,_ , _ = self._step(model, d, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config_pretrain.yaml", "r"), Loader=yaml.FullLoader)   
    print(config)

    from dataset.dataset import MoleculeDatasetWrapper

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    molclr = GCLmf(dataset, config)
    molclr.train()


if __name__ == "__main__":
    main()
