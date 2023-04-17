import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

from prefetch_generator import BackgroundGenerator

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(0,119))  #

BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]

def random_split(dataset):
    train_inds = []
    valid_inds = []
    test_inds = []
    
    for ind, smiles in enumerate(dataset.smiles_data):
        if dataset.data_group[ind] == 'test':
            test_inds.append(ind)
        elif dataset.data_group[ind] == 'training':
            train_inds.append(ind)
        else:
            valid_inds.append(ind)
            
    print('valid set:', len(valid_inds), 'test set:',len(test_inds))
    return train_inds, valid_inds, test_inds
    

def read_smiles(data_path, target, task):
    smiles_data, labels ,data_group= [], [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',') 
        for i, row in enumerate(csv_reader): 
            smiles = row['smiles']
            label = row[target]
            group = row['group']                                       
            mol = Chem.MolFromSmiles(smiles)
            if mol != None and label != '':
                smiles_data.append(smiles)
                data_group.append(group)
                if task == 'classification':
                    labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels, data_group


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.data_group = read_smiles(data_path, target, task)
        self.task = task

        self.conversion = 1

    def __getitem__(self, index): 
        mol = Chem.MolFromSmiles(self.smiles_data[index])   #self.smiles_data时一个列表，这里是取出对应index的分子
        mol = Chem.AddHs(mol)   #给分子加H


        type_idx = []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = x1
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BOND_LIST.index(bond.GetBondType())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BOND_LIST.index(bond.GetBondType())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task


    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        train_idx, valid_idx, test_idx = random_split(train_dataset)
        

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoaderX(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=False
        )
        valid_loader = DataLoaderX(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=False
        )
        test_loader = DataLoaderX(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=False
        )

        return train_loader, valid_loader, test_loader