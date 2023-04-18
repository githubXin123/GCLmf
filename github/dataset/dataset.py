import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

from queue import Queue
from threading import Thread


import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from prefetch_generator import BackgroundGenerator


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

ATOM_LIST = list(range(0,119))

BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]



def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)

    return smiles_data

#get the feature of the molecule
def get_feature(mol):
    type_idx = []

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    for atom in atoms:
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = x1
    x = torch.cat([x1, x2], dim = -1)
    
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

    data_00 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data_00


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)

    def __getitem__(self, index): 

        mol_number = index * 25
        mol_00 = Chem.MolFromSmiles(self.smiles_data[mol_number])

        data_00 = get_feature(mol_00)
        data_01 = data_00
        data_list = ['data_1','data_2','data_3','data_4','data_5','data_6',
                    'data_7','data_8','data_9','data_10','data_11','data_12',
                    'data_13','data_14','data_15','data_16','data_17','data_18',
                    'data_19','data_20','data_21','data_22','data_23','data_24',                  
                    ]
        
        mol_list = ['mol_1','mol_2','mol_3','mol_4','mol_5','mol_6',
                    'mol_7','mol_8','mol_9','mol_10','mol_11','mol_12',
                    'mol_13','mol_14','mol_15','mol_16','mol_17','mol_18',
                    'mol_19','mol_20','mol_21','mol_22','mol_23','mol_24',                  
                    ]
        
        for i in range(24):
            globals()[mol_list[i]] = Chem.AddHs(Chem.MolFromSmiles(self.smiles_data[mol_number+i+1]))
            globals()[data_list[i]] = get_feature(globals()[mol_list[i]])

        a = []
        a.append(data_00)
        a.append(data_01)
        for i in range(24):
            a.append(globals()[data_list[i]])

        return a

    def __len__(self):
        return len(self.smiles_data)
        
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset) // 25
        indices = list(range(num_train))
        
        random_state = np.random.RandomState(seed=888)
        random_state.shuffle(indices)


        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader =  DataLoaderX(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=True, persistent_workers=False)

        valid_loader =  DataLoaderX(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=True, persistent_workers=False)

        return train_loader, valid_loader
