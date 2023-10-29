import time
import argparse
import torch.optim as optim
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from data_preprocesing import load_data
from data_preprocesing import accuracy
import numpy as np

# from APPNP import APPNP_Layer
# from GDRA import GraphDistRecompAttentionLayer

class APPNP_Layer(nn.Module):
    def __init__(self, in_features, out_features, alpha, adj, iterations, dropout, device, model_mode="sparse"):
        super(APPNP_Layer, self).__init__()

        # Initialize the weights and setup the dropout layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.iterations = iterations
        self.dropout = dropout
        self.device = device
        self.model_mode = model_mode

        # The adjacency matrix remains fixed in the APPNP_ty layer.
        self.adj = adj

        # Weight initialization
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

        if(model_mode=="sparse"):
            # Extract indices where the adjacency matrix is non-zero
            non_zero_indices = torch.nonzero(adj).t()
            values = adj[non_zero_indices[0], non_zero_indices[1]]  # Extract the non-zero values
            self.adj = torch.sparse_coo_tensor(non_zero_indices, values, adj.shape)

    def forward(self, x):
        # Matrix multiplication
        support = torch.mm(x, self.weight)

        # Dropout (if training mode)
        support = F.dropout(support, self.dropout, training=self.training)

        # Personalized PageRank Approximation
        prev = support
        for _ in range(self.iterations):
            if(self.model_mode=="sparse"):
                support = torch.spmm(self.adj, support) * (1 - self.alpha) + prev * self.alpha
            else:
                support = torch.mm(self.adj, support) * (1 - self.alpha) + prev * self.alpha
            support = F.relu(support)

        return support


class GraphDistRecompAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, beta, eta, concat=True):
        super(GraphDistRecompAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta
        self.eta = eta
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.disc = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, lamda):
        # h代表(N, node_features) W为0线性变换层 用feature去预测class Wh中代表(in_nodes, out_class)
        # Wh代表了每个节点通过self.W线性变换之后，得到的7个不同的分数，然后对于7个不同的分数进行
        # 本质计算同GCN通过一步线性变化从每个节点到output
        # h is (in_nodes, node_features)
        # **self.W** is (node_features, out_class)
        # feature capturer: self.W from feature to a specific score [h*self.W]=Wh
        # feature correlation attention capturer: self.a to capture different attention weight from each output
        # and generate a N*N table that denote the node2node attention score
        # [[self.a[fh]*Wh]+[self.a[sh]*Wh^T]]

        global mask
        dischange = torch.zeros(h.shape[0], h.shape[0])
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        ones_vec = torch.ones(h.shape[0], h.shape[0])
        zeros_vec = torch.zeros(h.shape[0], h.shape[0])
        attention = torch.where(mask > 0, e, zero_vec)
        mask = torch.where(dischange > lamda, ones_vec, mask)  # ***替代方案是mask->cur_mask 每次更改某一些节点
        mask = torch.where(dischange < -lamda, zeros_vec, mask)
        attention = torch.where(mask > 0, e, zero_vec)  # 只要mask大于0，就标记了
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        dischange = self._compute_distance_change(Wh)
        dischange = dischange / (max(abs(dischange.min()), dischange.max()))
        Wh = torch.matmul(attention, Wh)

        if self.concat:
            # 使用激活函数
            return F.elu(Wh)
        else:
            # 直接返回主要的隐藏层
            return Wh

    def _prepare_attentional_mechanism_input(self, Wh):
        # 我认为attention table就是一个大小为N*N的矩阵，其中的每个元素代表i和j之间的相关关系
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # Wh1代表第一个out_feature层, Wh1大小为(N, 1)
        # (in_nodes,out_class)*(out_class, 1)
        # by attention we get a score for each nodes
        # 新的attention方法
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        # Wh2代表第二个out_feature层, Wh2大小为(N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        # 这里使用了广播机制其中两个(N,1)和(1,N)的向量相加得到(N,N)
        # 这是一个对称矩阵，其中矩阵中的每个数值的作用是学习两个节点之间的相关关系
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def _compute_distance_change(self, Wh):
        dv1 = torch.matmul(Wh, self.disc[:self.out_features, :])
        # Wh2代表第二个out_feature层, Wh2大小为(N, 1)
        dv2 = torch.matmul(Wh, self.disc[self.out_features:, :])
        # broadcast add
        # 这里使用了广播机制其中两个(N,1)和(1,N)的向量相加得到(N,N)
        # 这是一个对称矩阵，其中矩阵中的每个数值的作用是学习两个节点之间的相关关系
        dischange = dv1 + dv2.T
        return dischange  # self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    # 这个方程是一个稀疏矩阵的乘法
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    # 前向传播，调用函数进行运算
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        # torch.sparse
        a = torch.sparse_coo_tensor(indices, values, shape)
        # ctx代表
        ctx.save_for_backward(a, b)
        # ctx
        ctx.node_num = shape[0]
        return torch.matmul(a, b)

    # 反向传播，使用梯度进行计算
    @staticmethod
    def backward(ctx, grad_output):
        # a, b代表tensor值
        a, b = ctx.saved_tensors
        # grad_values代表梯度值
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.node_num + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class APPNP_GDRA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, beta, eta, lamda, adj, nheads, iterations, device,
                 model_mode):
        super(APPNP_GDRA, self).__init__()

        self.dropout = dropout
        self.lamda = lamda

        # Incorporating GraphDistRecompAttentionLayer
        self.gdra = GraphDistRecompAttentionLayer(nfeat, nhid, dropout, alpha, beta, eta)

        # Creating multiple attention layers for APPNP
        self.attentions = [
            APPNP_Layer(nhid, nhid, alpha=alpha, adj=adj, iterations=iterations, dropout=dropout, device=device,
                        model_mode=model_mode) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Output layer
        self.out_att = APPNP_Layer(nhid * nheads, nclass, alpha=alpha, adj=adj, iterations=iterations, dropout=dropout,
                                   device=device, model_mode=model_mode)

    def forward(self, x):
        # Applying GraphDistRecompAttentionLayer first
        x = self.gdra(x, self.lamda)

        # Apply dropout and then APPNP_ty layers
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)

        # Apply dropout and then the output layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))

        return F.log_softmax(x, dim=1)
