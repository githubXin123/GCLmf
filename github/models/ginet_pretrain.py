import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


'''修改1'''
num_atom_type = 119 # 原始模型为119（原子类型118+mask掉的原子），而我也是119（原子类型118+分子片段断裂处的原子）

num_bond_type = 5 # including aromatic and self-loop edge，和原始模型保持一致，5（4中键类型+自环）




'''修改2'''
class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(),       #原始模型为nn.ReLU()
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_index.device)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):  
        return x_j + edge_attr  #x_j的形状[2M+N,300],edge_attr的形状也是[2M+N,300]

    def update(self, aggr_out): #aggr_out应该为消息传递、聚合后得到的结果
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)    #nn.Embedding()其为一个简单的存储固定大小的词典的嵌入向量的查找表
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList() #nn.ModuleList()可以以列表的形式来保持多个子模块。
        for layer in range(num_layer):  
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):  #批标准化
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add': #即求和
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        
        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),   #原始模型为nn.ReLU(inplace=True)
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )


    def forward(self, data):
    #data由四部分构成：原子数data.batch，原子信息data.x，边索引data.edge_index，边属性data.edge_attr;
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0])

        for layer in range(self.num_layer):            
            h = self.gnns[layer](h, edge_index, edge_attr)  #卷积操作 
            h = self.batch_norms[layer](h)
            #config文件中的deop率为0相当于不进行dropout
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training) 
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
         
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
       
        out = self.out_lin(h)
        
        #添加噪音的位置，在latent space添加
        # if noise == True:
            # for i in range(5):
                # 高斯分布噪声
                # random_noise = torch.randn(out.shape).to(edge_index.device)
                # 均匀分布噪声
                # random_noise = torch.FloatTensor(out.shape).uniform_(0, 1).to(edge_index.device)
                # out_sign = torch.sign(out)
                # out += torch.mul(out_sign,F.normalize(random_noise, dim=1, p=2)) * 0.1
                
        return h, out
