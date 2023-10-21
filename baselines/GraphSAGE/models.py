import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set
        if not num_sample is None:
            # num_sample是m+
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
            # samp_neighs是[{1,2,3},{3,5},{6,7}] 射出点的集合

        # 如果是GCN的话添加自环
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        # 一个点在一层被多次采样到，也只考虑一次的效果
        unique_nodes_list = list(set.union(*samp_neighs))  # 这里代表射出点的unique_nodes
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)

        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        # if gcn: (embed_dim, feat_dim)
        # if not gcn: (embed_dim, 2 * feat_dim)
        init.xavier_uniform_(self.weight)

        # print(self.features) #Embedding(2708, 1433)
        # print(self.feat_dim) #1433
        # print(self.adj_lists) #adj_list {src_node:{to_nodes}}
        # print(self.aggregator) #MeanAggregator()
        # print(self.num_sample) #10
        # print(base_model)
        """
        Encoder(
          (features): Embedding(2708, 1433)
          (aggregator): MeanAggregator(
            (features): Embedding(2708, 1433)
          )
        )
        """
        # print(self.gcn) #True
        # print(self.embed_dim) #128
        # print(self.cuda) #False
        # print("aggregator: ", self.aggregator) #aggregator:  MeanAggregator()
        # print("weight: ", self.weight)
        """
        tensor([[-0.1422, -0.0802,  0.0455,  ..., -0.0922,  0.0607, -0.1466],
        [ 0.0816, -0.1112, -0.0079,  ...,  0.0417, -0.0330, -0.0945],
        [ 0.0754, -0.0852, -0.1137,  ..., -0.1153, -0.0737, -0.1089],
        ...,
        [ 0.0341, -0.0400, -0.0973,  ...,  0.0266, -0.1189,  0.0885],
        [-0.1442, -0.1436,  0.0275,  ..., -0.0916, -0.0287, -0.0882],
        [-0.0701,  0.0136,  0.0744,  ...,  0.0113, -0.0616,  0.0938]],
       requires_grad=True)
        """
        # print("weight.shape: ", self.weight.shape) #weight.shape:  torch.Size([128, 128])

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes 这里的nodes是待采样下一层的本层点
        """

        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)
        # print("neigh_feat", neigh_feats.shape)
        """
        num_neigh.shape:  torch.Size([724, 1])
        neigh_feat torch.Size([724, 1433])
        neigh_feat torch.Size([256, 128])
        """
        """
        num_neigh.shape:  torch.Size([677, 1])
        neigh_feat torch.Size([677, 1433])
        neigh_feat torch.Size([256, 128])
        """
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
                # 从feature大矩阵中select到本层节点所对应的feature
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
            # 1为横向,0为纵向
        else:
            combined = neigh_feats
        # print("A:: ",self.weight.shape)
        # print("B:: ",combined.t().shape)
        combined = F.relu(self.weight.mm(combined.t()))  # combined.t() * self.weight

        # 如何得到self.weight, combined

        # print("combined:: ", combined.shape)
        """
        neigh_feat torch.Size([694, 1433])
        A::  torch.Size([128, 1433])
        B::  torch.Size([694, 128])
        neigh_feat torch.Size([256, 128])
        A::  torch.Size([128, 128])
        B::  torch.Size([256, 128])
        """

        return combined

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, adj_lists, feat_data, num_samples=5):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_lists = adj_lists

        self.features = nn.Embedding(feat_data.shape[0], input_dim)
        self.features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

        self.agg_layers = []
        self.enc_layers = []
        for i in range(num_layers):
            if i == 0:
                agg = MeanAggregator(self.features, cuda=True)
                enc = Encoder(self.features, input_dim, output_dim, adj_lists, agg, gcn=True, cuda=False)
            else:
                agg = MeanAggregator(lambda nodes, enc_index=i: self.enc_layers[enc_index - 1](nodes).t(), cuda=False)
                enc = Encoder(lambda nodes, enc_index=i: self.enc_layers[enc_index - 1](nodes).t(), output_dim,
                              output_dim, adj_lists, self.agg_layers[-1], base_model=self.enc_layers[-1], gcn=True,
                              cuda=False)

            enc.num_samples = num_samples
            self.agg_layers.append(agg)
            self.enc_layers.append(enc)

        for i, enc in enumerate(self.enc_layers):
            self.add_module('enc_{}'.format(i), enc)

    def forward(self):
        return self.enc_layers[-1]#(nodes)
