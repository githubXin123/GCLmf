import torch
import numpy as np
import random

class NTXentLoss(torch.nn.Module):
    #Normalized Temperature-scaled Cross Entropy(NCE)
    def __init__(self, device, batch_size, temperature, use_cosine_similarity, para_1 = 1):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.mask_samples_from_same_repr_1 = self._get_correlated_mask(type_mask = 1).type(torch.bool)
        
        self.mask_samples_from_same_repr_2 = self._get_correlated_mask(type_mask = 2).type(torch.bool)

        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.para_1 = para_1
        self.repre_list = ['repre_1','repre_2','repre_3','repre_4','repre_5','repre_6',
                            'repre_7','repre_8','repre_9','repre_10','repre_11','repre_12']
        self.simi_list = ['simi_1','simi_2','simi_3','simi_4','simi_5','simi_6',
                            'simi_7','simi_8','simi_9','simi_10','simi_11','simi_12']
        self.nega_list = ['nega_1','nega_2','nega_3','nega_4','nega_5','nega_6',
                            'nega_7','nega_8','nega_9','nega_10','nega_11','nega_12']
        

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, type_mask = 1):
        diag = np.eye(2 * self.batch_size) 
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size) 
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        
        mask = torch.from_numpy((diag + l1 + l2))
        
        
        if type_mask ==1:
            mask = mask.type(torch.bool)
            return mask.to(self.device)
        else:
            mask = (1 - mask).type(torch.bool)
            return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)    
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, loss_embedding, n_iter):
    
        repre_0 = torch.cat([loss_embedding[0], loss_embedding[1]], dim=0)
        
        for i in range(0,24,2):
            globals()[self.repre_list[i//2]] = torch.cat([repre_0, loss_embedding[i+2], loss_embedding[i+3]], dim=0)
        
        simi_0 = self.similarity_function(repre_0, repre_0)
        
        for i in range(12):
            globals()[self.simi_list[i]] = self.similarity_function(globals()[self.repre_list[i]], globals()[self.repre_list[i]])
        
        l_pos = torch.diag(simi_0, self.batch_size)
        r_pos = torch.diag(simi_0, -self.batch_size)
        
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        for i in range(12):
            globals()[self.simi_list[i]] = globals()[self.simi_list[i]][:2 * self.batch_size,2 * self.batch_size:]
        
        for i in range(12):
            globals()[self.nega_list[i]] = globals()[self.simi_list[i]][self.mask_samples_from_same_repr_1].view(2 * self.batch_size, -1)

        nega_hard = torch.cat([nega_1,nega_2,nega_3,nega_4,nega_5,nega_6,nega_7,nega_8,nega_9,nega_10,nega_11,nega_12], dim = 1)

        nega_raw = simi_0[self.mask_samples_from_same_repr_2].view(2 * self.batch_size, -1)
        
        random_num = random.randint(0, nega_raw.shape[1]-24)
        nega_raw1 = nega_raw[:,:random_num]
        nega_raw2 = nega_raw[:,random_num+24:]
        nega_raw = torch.cat((nega_raw1, nega_raw2), dim = 1)
        
        nega_sum = torch.cat((nega_raw, nega_hard), dim = 1)

        logits = torch.cat((positives, nega_sum), dim=1)
        logits /= self.temperature
        

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels) / (2 * self.batch_size)
            
        return loss
