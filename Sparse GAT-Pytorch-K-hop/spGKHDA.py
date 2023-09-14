import torch
import torch.nn as nn
import torch.nn.functional as F
from spGKHDA_Layer import spGraphKHopDecentDiffAttentionLayer
from sparse_multi_hop_sampling import preprocess_khop

class spGKHDA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, k, beta, num_sample, adj, theta):
        """Dense version of GAT."""
        """
            nfeat= 1433
            nhid= 8
            nclass= 7
            dropout= 0.6
            nheads= 8
            alpha= 0.2
        """
        super(spGKHDA, self).__init__()
        self.dropout = dropout

        self.attentions = [
            spGraphKHopDecentDiffAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, hop_num=k, beta=beta,
                                                concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = spGraphKHopDecentDiffAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                                           hop_num=k, beta=beta, concat=False)

        edge = adj.nonzero().t()  # edge: torch.Size([2, 108365])
        self.k_edge_list = preprocess_khop(edge, k, len(adj), theta)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)  # attention elem shape torch.Size([2708, 8])

        # 注意这里是聚合1layer的feature
        print(x.shape)
        # print(self.k_edge_list)
        print(self.attentions[0](x, self.k_edge_list))
        x = torch.cat([att(x, self.k_edge_list) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.out_att(x, self.k_edge_list))

        return F.log_softmax(x, dim=1)