import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from preprocessing import preprocess_khop
from GKHDA_layer import GraphKHopDecentDiffAttentionLayer


class GKHDA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,k,beta, num_sample, adj):
        """Dense version of GAT."""
        """
            nfeat= 1433
            nhid= 8
            nclass= 7
            dropout= 0.6
            nheads= 8
            alpha= 0.2
        """
        super(GKHDA, self).__init__()
        self.dropout = dropout
        
        
        # nfeat--input_features 1433, nhid--7 nclass, dropout, alpha, nheadsnheads=8
        # 这里可以理解为每个self.attention是由多个GAT layer构成的，每个独立算一组权重self.W和self.a
        self.attentions = [GraphKHopDecentDiffAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha,k=k,beta=beta, concat=True) for _ in range(nheads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphKHopDecentDiffAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, k=k,beta=beta, concat=False)
        
        self.k_mask_list=preprocess_khop(adj, k,  num_sample)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        """
        print("attention 0 elem",self.attentions[0](x,adj))
        print("attention elem shape",self.attentions[0](x,adj).shape)
        """
        """
        attention 0 elem tensor([[-0.0068,  0.0113,  0.0025,  ..., -0.0331,  0.0052, -0.0216],
                                [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                                [-0.0109,  0.0117,  0.0145,  ...,  0.0121,  0.0081, -0.0068],
                                ...,
                                [-0.0173, -0.0013, -0.0149,  ..., -0.0122, -0.0033,  0.0425],
                                [ 0.0059,  0.0022,  0.0123,  ...,  0.0020, -0.0051,  0.0057],
                                [ 0.0128, -0.0093,  0.0293,  ...,  0.0104,  0.0174,  0.0023]],
                               grad_fn=<EluBackward0>)
        attention elem shape torch.Size([2708, 8])
        """
        
        #注意这里是聚合1layer的feature
        print(x.shape)
        print(self.k_mask_list)
        print(self.attentions[0](x,self.k_mask_list))
        x = torch.cat([att(x,self.k_mask_list) for att in self.attentions], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.elu(self.out_att(x,self.k_mask_list))
        
        return F.log_softmax(x, dim=1)