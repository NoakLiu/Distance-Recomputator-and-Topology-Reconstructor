"""
https://github.com/Diego999/pyGAT/blob/master/layers.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable


class GraphKHopDecentDiffAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, k, beta, concat=True):
        super(GraphKHopDecentDiffAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 
        self.beta = beta
        self.hop_num=k
        # concat我认为是连接多头的学习，concat=True为多头，False为单头
        # self.W的大小是(in_features, out_features)
        # 也就是 output=W*F (F matrix is the feautre matrix for each node; W is a n*n matrix)
        # torch.empty是没有定义的随机初始化代码
        self.W = nn.Parameter(torch.empty(size=(in_features*k, out_features)))
        #这里代表把W数据初始化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a代表attention 其中为长度为2*out_feature的初始化程度，这里代表对于feature中的不同位置的attention层
        # 我认为此处的attention会在之后乘积方式计算
        # attention一开始随机初始化初始化，之后每次对于out feature(7)不同的feature进行不同的weight学习
        # Graph Attention Layer中每次对于out feature(7)中不同的位置进学习
        self.a = nn.Parameter(torch.empty(size=(2*out_features*k, 1)))
        # xavier_uniform_代表self.a.data
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        #LeakyReLu泄露版ReLU就是小于0的时候是y=k*x，大于0的时候是y=x
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        """
        self.mask_list=[]
        for i in range(0,self.hop_num):
            samp_neighs_for_khop=sample_around_for_khop(nodes,i+1, self.num_sample,adj)
            # print(samp_neighs_for_khop)
            column_indices = list(set( [samp_neigh for samp_neighs in samp_neighs_for_khop for samp_neigh in samp_neighs]))
            mask = Variable(torch.zeros(adj.shape))
            print("col: ",len(column_indices))
            print("row: ",len(row_indices))
            mask[row_indices, column_indices] = 1
            self.mask_list.append(mask)
        print("preprocess end")
        """
            

    def forward(self, h, k_mask_list):
        # h代表(N, node_features) W为0线性变换层 用feature去预测class Wh中代表(in_nodes, out_class)
        # Wh代表了每个节点通过self.W线性变换之后，得到的7个不同的分数，然后对于7个不同的分数进行
        # 本质计算同GCN通过一步线性变化从每个节点到output
        # h is (in_nodes, node_features)
        # **self.W** is (node_features, out_class)
        # feature capturer: self.W from feature to a specific score [h*self.W]=Wh
        # feature correlation attention capturer: self.a to capture different attention weight from each output 
        # and generate a N*N table that denote the node2node attention score
        # [[self.a[fh]*Wh]+[self.a[sh]*Wh^T]]       
        
        for i in range(self.hop_num-1,-1,-1):
            print(i)
            # 我认为的改进：增加隐藏层去更好表示
            # attention table是否可以rep2rep而不是node2node
            # attention合用一段attention
            # 多阶段attention并用
            Whnew = torch.mm(h, self.W[i*self.in_features:(i+1)*self.in_features,:]) 
            if(i==(self.hop_num-1)):
                Wh=Whnew
            # h.shape: (N, in_features), Wh.shape: (N, out_features), self.W.shape: (in_features, out_features)
            e = self._prepare_attentional_mechanism_input(Whnew,i)

            # zero vec代表的是这里每个元素被初始化为非常小的数字
            zero_vec = -9e15*torch.ones_like(e) 
            # attention中的torch被初始化为：如果adj此处两点相连接，就设置为
            # attention中的torch是非常重要的
            # print(adj.shape)
            # print(adj.dtype)
            """
            samp_neighs_for_khop=sample_around_for_khop(nodes,i+1, self.num_sample,adj)
            # print(samp_neighs_for_khop)
            column_indices = list(set( [samp_neigh for samp_neighs in samp_neighs_for_khop for samp_neigh in samp_neighs]))
            mask = Variable(torch.zeros(adj.shape))
            print("col: ",len(column_indices))
            print("row: ",len(row_indices))
            mask[row_indices, column_indices] = 1
            """
            """
            if self.cuda:
                mask = mask.cuda()
            """
            cur_mask=k_mask_list[i]
            attention = torch.where(cur_mask > 0, e, zero_vec)
            # attention中的softmax被初始化为attention, dim=1
            attention = F.softmax(attention, dim=1)
            # attention中的dropout代表这个函数的dropout, 其中attention, dropout, training需要合连
            # print("training=",self.training) #True
            attention = F.dropout(attention, self.dropout, training=self.training)
            # h_prime代表主要的隐藏层
            Wh = (1-self.beta)*torch.matmul(attention, Whnew)+self.beta*Wh #h为隐藏层
            
            if(i==0):
                if self.concat:
                    # 使用激活函数
                    return F.elu(Wh)
                else:
                    # 直接返回主要的隐藏层
                    return Wh

    def _prepare_attentional_mechanism_input(self, Wh, hop_num):
        # 我认为attention table就是一个大小为N*N的矩阵，其中的每个元素代表i和j之间的相关关系
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # Wh1代表第一个out_feature层, Wh1大小为(N, 1)
        # (in_nodes,out_class)*(out_class, 1)
        # by attention we get a score for each nodes
        # 新的attention方法
        Wh1 = torch.matmul(Wh, self.a[(2*hop_num)*self.out_features:(2*hop_num+1)*self.out_features, :])
        # Wh2代表第二个out_feature层, Wh2大小为(N, 1)
        Wh2 = torch.matmul(Wh, self.a[(2*hop_num+1)*self.out_features:(2*hop_num+2)*self.out_features, :])
        # broadcast add
        # 这里使用了广播机制其中两个(N,1)和(1,N)的向量相加得到(N,N)
        # 这是一个对称矩阵，其中矩阵中的每个数值的作用是学习两个节点之间的相关关系
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

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
        ctx.N = shape[0]
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
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b