#!/usr/bin/env python
# coding: utf-8

# Sampling Module for dense

# In[ ]:


nodes_to_nghs={}
def single_node_sample_around(center_node, num_sample, adj_lists):
    """
    Input:
        center_node: node_id 5
        num_sample: 3
    Output:
        list of output nodes: {6,3,9}
    """
    _set = set
    _sample = random.sample
    if(center_node in nodes_to_nghs):
        if(num_sample==len(nodes_to_nghs[center_node])):
            return nodes_to_nghs[center_node]
        elif(num_sample<len(nodes_to_nghs[center_node])):
            return _set(_sample(nodes_to_nghs[center_node],num_sample,))
    nghs=[]
    for i in range(0,len(adj_lists[center_node])):
        if(adj_lists[int(center_node)][i]!=0):
            nghs.append(i)
    nghs=_set(nghs) #{6,3,0,9,11}
    # print(nghs)
    if len(nghs)>=num_sample:
        samp_neighs=_set(_sample(nghs,num_sample,))
    else:
        samp_neighs=nghs
    #{6,3,9}
    if(center_node not in nodes_to_nghs):
        nodes_to_nghs[center_node]=samp_neighs
    else:
        if(len(nodes_to_nghs[center_node])<len(samp_neighs)):
            nodes_to_nghs[center_node]=samp_neighs
    return samp_neighs


# In[ ]:


def sample_around_for_khop(center_nodes, k, num_sample,adj_lists):
    """
    Return the kth layer neighborhood
    Input:
        center_nodes: a list of center nodes that to sample [0,1,..,2707]
        number_sample: the total number select from the next layer of the center nodes
        {5,2,3}->5:3,2:4,3:3
        {2,3}->2:5,3:5
    Output:
        [{6,3,9},{4,2,9,6},{2,1}]
    """
    center_nodes = [{center_node} for center_node in center_nodes]
    #print(center_nodes)
    for h in range(0,k):
        per_hop_nghs=[]
        for i in range(0,len(center_nodes)):
            cnls=list(center_nodes[i])
            ln=len(cnls)
            rand_ls=[random.randint(1,100) for j in range(0,ln)]
            rand_ls_sum=sum(rand_ls)
            p_ls=list(map(lambda i:i/rand_ls_sum,rand_ls))
            sample_num_ls=[elem*num_sample for elem in p_ls]
            nx_level_elem=set()
            for j in range(0,ln):
                nx_level_elem=nx_level_elem.union(single_node_sample_around(cnls[j],int(sample_num_ls[j]),adj_lists))
                #print(single_node_sample_around(cnls[j],int(sample_num_ls[j]),adj_lists))
            #print(nx_level_elem)
            per_hop_nghs.append(list(nx_level_elem))
        center_nodes=per_hop_nghs
        if(h==k-1):
            return per_hop_nghs


# In[ ]:


def preprocess_khop(adj, hop_num,  num_sample):
    print("preprocess begin")
    nodes=[rowid for rowid in range(0,len(adj))]
    #row_indices = nodes #one node->one_node
    mask_list=[]
    for i in range(0,hop_num):
        samp_neighs_for_khop=sample_around_for_khop(nodes,i+1, num_sample,adj)
        #column_indices=sample_around_for_khop(nodes,i+1, num_sample,adj)
        # print(samp_neighs_for_khop)
        row_indices = [i for i in range(len(samp_neighs_for_khop)) for samp_neigh in samp_neighs_for_khop[i]]
        column_indices = [samp_neigh for samp_neighs in samp_neighs_for_khop for samp_neigh in samp_neighs]
        mask = Variable(torch.zeros(adj.shape))
        print("col: ",len(column_indices))
        print("row: ",len(row_indices))
        mask[row_indices, column_indices] = 1
        mask_list.append(mask)
    print("preprocess end")
    return mask_list


# Sampling Method for Sparse

# In[ ]:





# In[ ]:





# K-hop Propagation Implementation

# In[ ]:


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
        self.W = nn.Parameter(torch.empty(size=(in_features*k, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features*k, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
            

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
            Whnew = torch.mm(h, self.W[i*self.in_features:(i+1)*self.in_features,:]) 
            if(i==(self.hop_num-1)):
                Wh=Whnew
            e = self._prepare_attentional_mechanism_input(Whnew,i)
            zero_vec = -9e15*torch.ones_like(e) 
            cur_mask=k_mask_list[i]
            attention = torch.where(cur_mask > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            Wh = (1-self.beta)*torch.matmul(attention, Whnew)+self.beta*Wh #h为隐藏层
            
            if(i==0):
                if self.concat:
                    return F.elu(Wh)
                else:
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
        Wh2 = torch.matmul(Wh, self.a[(2*hop_num+1)*self.out_features:(2*hop_num+2)*self.out_features, :])
        # broadcast
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# In[ ]:


class SpecialSpmmFunction(torch.autograd.Function):
    # 这个方程是一个稀疏矩阵的乘法
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    # 前向传播，调用函数进行运算
    def forward(ctx, indices, values, shape, b): #forward return computation result
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    # 反向传播，使用梯度进行计算
    @staticmethod
    def backward(ctx, grad_output): #backward return loss gradient
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

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


# In[ ]:


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        print("input:",input)
        print("input.shape:",input.shape)
        
        """
        input: tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0285, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]])
        
        input.shape: torch.Size([19717, 500])
        """

        N = input.size()[0] #N是节点数
        print("N:",N) 
        #19717
        edge = adj.nonzero().t() 
        #edge是adj中非零元素
        #edge: tensor([[    0,     0,     0,  ..., 19715, 19716, 19716],
        #[    0,  1378,  1544,  ..., 19715, 16030, 19716]])
        print("edge:",edge)
        print("edge.shape:",edge.shape)
        #这里的edge是边的集合，由一个点指向另一个点，使用边进行指向
        #edge: torch.Size([2, 108365])
        """
        edge: tensor([[    0,     0,     0,  ..., 19715, 19716, 19716],
        [    0,  1378,  1544,  ..., 19715, 16030, 19716]])
        edge.shape: torch.Size([2, 108365])
        """

        h = torch.mm(input, self.W) #h是input-->W的变换
        """
        (N,feature)-->(N, output) 学习feature，考虑到每个点在feature的共通性，用一个(feature,output)的矩阵
        """
        print("h:",h)
        print("h.shape:",h.shape)
        """
        h: tensor([[-0.0122, -0.0181, -0.0114,  ...,  0.0088, -0.0128, -0.0391],
        [-0.0076,  0.0035, -0.0405,  ..., -0.0018, -0.0423,  0.0274],
        [-0.0389,  0.0143,  0.0055,  ...,  0.0151, -0.0111,  0.0050],
        ...,
        [ 0.0102,  0.0135,  0.0024,  ..., -0.0327, -0.0150, -0.0290],
        [-0.0243,  0.0269,  0.0032,  ...,  0.0173,  0.0452,  0.0024],
        [ 0.0126,  0.0079, -0.0386,  ..., -0.0229, -0.0021, -0.0229]],
       grad_fn=<MmBackward0>)
    
        h.shape: torch.Size([19717, 8])
        """

        # h.shape: torch.Size([19717, 8])
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() #edge是从头到尾转移
        # h: N x out; edge: 2 x E; h[edge[0,:],:]: E x out; edge_h: E x 2*out; edge_h (t): 2*out x E
        print("edge_h:",edge_h)
        print("edge_h.shape:",edge_h.shape)
        """
        edge_h: tensor([[-0.0122, -0.0122, -0.0122,  ..., -0.0243,  0.0126,  0.0126],
        [-0.0181, -0.0181, -0.0181,  ...,  0.0269,  0.0079,  0.0079],
        [-0.0114, -0.0114, -0.0114,  ...,  0.0032, -0.0386, -0.0386],
        ...,
        [ 0.0088,  0.0266, -0.0143,  ...,  0.0173,  0.0013, -0.0229],
        [-0.0128, -0.0098,  0.0087,  ...,  0.0452, -0.0112, -0.0021],
        [-0.0391,  0.0057, -0.0050,  ...,  0.0024, -0.0082, -0.0229]],
       grad_fn=<TBackward0>)
        
        edge_h.shape: torch.Size([16, 108365])
        """
        # edge_h.shape: torch.Size([16, 108365])
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze())) #edge是
        # a: 1 x 2*out
        # edge_h: 2*out x E
        # edge_e: 1 x E -->E
        # 每个边凝缩为一个数表示，每个v->v的点对用attention计算成一个数字，是edge(adj)的矩阵的值
        # self.a 与 edge_h的关系
        print("edge_e:",edge_e)
        print("edge_e.shape:",edge_e.shape)
        """
        edge_e: tensor([1.0011, 0.9814, 0.9661,  ..., 1.0043, 0.9730, 0.9576],
       grad_fn=<ExpBackward0>)
        
        edge_e.shape: torch.Size([108365])
        """
        
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # 2 x E(s)--> N x N(d); all ones: N x 1(d)
        print("e_rowsum:",e_rowsum) #每行求和
        print("e_rowsum.shape:",e_rowsum.shape)
        # e_rowsum: N x 1
        # e_rowsum每个点连接的边的数目

        edge_e = self.dropout(edge_e)
        print("edge_e:", edge_e)
        print("edge_e.shape", edge_e.shape)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # 2 x E(s)--> N x N(d); h(d):N x out
        # 考虑一节拓扑结构，聚合一阶拓扑结果（是由本身的点）
        print("h_prime:",h_prime)
        print("h_prime.shape:",h_prime.shape)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum) 
        # 这里考虑到拓扑结构，使用h_prime/e_rowsum得到矩阵的和
        # h_prime/e_rowsum
        print("h_prime:",h_prime)
        print("h_prime.shape:",h_prime.shape)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        
        #D^-1*A(value of GAT)

        if self.concat:
            # if this layer is not last layer
            res = F.elu(h_prime)
            print("return-notlast:", res)
            print("return-notlast.shape:",res.shape)
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            print("return-last:",h_prime)
            print("return-last.shape:", h_prime.shape)
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Original Method

# In[1]:


"""
https://github.com/Diego999/pyGAT/blob/master/layers.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b): #计算所需值 matrix(indices,values,shape)*b
        assert indices.requires_grad == False
        print("ctx:",ctx)
        print("ctx.shape:",ctx)
        print("indices:",indices)
        print("indices.shape:",indices.shape)
        print("values:",values)
        print("values.shape:",values.shape)
        print("shape:",shape)
        print("b.shape:",b.shape)
        """
        ctx: <torch.autograd.function.SpecialSpmmFunctionBackward object at 0x0000026CFFAC2B88>
        ctx.shape: <torch.autograd.function.SpecialSpmmFunctionBackward object at 0x0000026CFFAC2B88>
        indices: tensor([[    0,     0,     0,  ..., 19715, 19716, 19716],
                         [    0,  1378,  1544,  ..., 19715, 16030, 19716]])
        indices.shape: torch.Size([2, 108365])
        values: tensor([2.4262, 0.0000, 0.0000,  ..., 2.3776, 2.3106, 0.0000],
               grad_fn=<MulBackward0>)
        values.shape: torch.Size([108365])
        shape: torch.Size([19717, 19717])
        b.shape: torch.Size([19717, 8])
        """
        
        a = torch.sparse_coo_tensor(indices, values, shape) #return a sparse matrix
        print("a tosparse shape:",a.shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output): #计算所需值：
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            print("grad_a_dense:",grad_a_dense)
            print("grad_a_dense.shape:",grad_a_dense.shape)
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            print("edge_idx:",edge_idx)
            print("edge_idx.shape:",edge_idx.shape)
            grad_values = grad_a_dense.view(-1)[edge_idx]
            print("grad_values:",grad_values)
            print("grad_values.shape:",grad_values.shape)
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
            print("grad_b:",grad_b)
            print("grad_b.shape:",grad_b.shape)
        """
        grad_a_dense: tensor([[-1.6442e-05, -9.8595e-05, -1.4123e-04,  ..., -4.0126e-04,
          1.8964e-04, -1.5959e-04],
        [-3.3113e-05, -1.6949e-04, -2.4589e-04,  ..., -7.0223e-04,
          3.3010e-04, -2.8006e-04],
        [-2.4706e-04,  3.7611e-04,  2.1994e-04,  ...,  3.3569e-04,
         -3.2321e-04,  6.2765e-05],
        ...,
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]])
        grad_a_dense.shape: torch.Size([19717, 19717])
        edge_idx: tensor([        0,      1378,      1544,  ..., 388740370, 388756402,
                388760088])
        edge_idx.shape: torch.Size([108365])
        grad_values: tensor([-1.6442e-05,  4.2766e-05, -2.2224e-06,  ...,  0.0000e+00,
                 0.0000e+00,  0.0000e+00])
        grad_values.shape: torch.Size([108365])
        grad_b: tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                ...,
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
        grad_b.shape: torch.Size([19717, 3])
        """
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        print("input:",input)
        print("input.shape:",input.shape)
        
        """
        input: tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0285, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]])
        
        input.shape: torch.Size([19717, 500])
        """

        N = input.size()[0] #N是节点数
        print("N:",N) 
        #19717
        edge = adj.nonzero().t() 
        #edge是adj中非零元素
        #edge: tensor([[    0,     0,     0,  ..., 19715, 19716, 19716],
        #[    0,  1378,  1544,  ..., 19715, 16030, 19716]])
        print("edge:",edge)
        print("edge.shape:",edge.shape)
        #这里的edge是边的集合，由一个点指向另一个点，使用边进行指向
        #edge: torch.Size([2, 108365])
        """
        edge: tensor([[    0,     0,     0,  ..., 19715, 19716, 19716],
        [    0,  1378,  1544,  ..., 19715, 16030, 19716]])
        edge.shape: torch.Size([2, 108365])
        """

        h = torch.mm(input, self.W) #h是input-->W的变换
        """
        (N,feature)-->(N, output) 学习feature，考虑到每个点在feature的共通性，用一个(feature,output)的矩阵
        """
        print("h:",h)
        print("h.shape:",h.shape)
        """
        h: tensor([[-0.0122, -0.0181, -0.0114,  ...,  0.0088, -0.0128, -0.0391],
        [-0.0076,  0.0035, -0.0405,  ..., -0.0018, -0.0423,  0.0274],
        [-0.0389,  0.0143,  0.0055,  ...,  0.0151, -0.0111,  0.0050],
        ...,
        [ 0.0102,  0.0135,  0.0024,  ..., -0.0327, -0.0150, -0.0290],
        [-0.0243,  0.0269,  0.0032,  ...,  0.0173,  0.0452,  0.0024],
        [ 0.0126,  0.0079, -0.0386,  ..., -0.0229, -0.0021, -0.0229]],
       grad_fn=<MmBackward0>)
    
        h.shape: torch.Size([19717, 8])
        """

        # h.shape: torch.Size([19717, 8])
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() #edge是从头到尾转移
        # h: N x out; edge: 2 x E; h[edge[0,:],:]: E x out; edge_h: E x 2*out; edge_h (t): 2*out x E
        print("edge_h:",edge_h)
        print("edge_h.shape:",edge_h.shape)
        """
        edge_h: tensor([[-0.0122, -0.0122, -0.0122,  ..., -0.0243,  0.0126,  0.0126],
        [-0.0181, -0.0181, -0.0181,  ...,  0.0269,  0.0079,  0.0079],
        [-0.0114, -0.0114, -0.0114,  ...,  0.0032, -0.0386, -0.0386],
        ...,
        [ 0.0088,  0.0266, -0.0143,  ...,  0.0173,  0.0013, -0.0229],
        [-0.0128, -0.0098,  0.0087,  ...,  0.0452, -0.0112, -0.0021],
        [-0.0391,  0.0057, -0.0050,  ...,  0.0024, -0.0082, -0.0229]],
       grad_fn=<TBackward0>)
        
        edge_h.shape: torch.Size([16, 108365])
        """
        # edge_h.shape: torch.Size([16, 108365])
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze())) #edge是
        # a: 1 x 2*out
        # edge_h: 2*out x E
        # edge_e: 1 x E -->E
        # 每个边凝缩为一个数表示，每个v->v的点对用attention计算成一个数字，是edge(adj)的矩阵的值
        # self.a 与 edge_h的关系
        print("edge_e:",edge_e)
        print("edge_e.shape:",edge_e.shape)
        """
        edge_e: tensor([1.0011, 0.9814, 0.9661,  ..., 1.0043, 0.9730, 0.9576],
       grad_fn=<ExpBackward0>)
        
        edge_e.shape: torch.Size([108365])
        """
        
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # 2 x E(s)--> N x N(d); all ones: N x 1(d)
        print("e_rowsum:",e_rowsum) #每行求和
        print("e_rowsum.shape:",e_rowsum.shape)
        # e_rowsum: N x 1
        # e_rowsum每个点连接的边的数目

        edge_e = self.dropout(edge_e)
        print("edge_e:", edge_e)
        print("edge_e.shape", edge_e.shape)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # 2 x E(s)--> N x N(d); h(d):N x out
        # 考虑一节拓扑结构，聚合一阶拓扑结果（是由本身的点）
        print("h_prime:",h_prime)
        print("h_prime.shape:",h_prime.shape)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum) 
        # 这里考虑到拓扑结构，使用h_prime/e_rowsum得到矩阵的和
        # h_prime/e_rowsum
        print("h_prime:",h_prime)
        print("h_prime.shape:",h_prime.shape)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        
        #D^-1*A(value of GAT)

        if self.concat:
            # if this layer is not last layer
            res = F.elu(h_prime)
            print("return-notlast:", res)
            print("return-notlast.shape:",res.shape)
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            print("return-last:",h_prime)
            print("return-last.shape:", h_prime.shape)
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Multi-Hop

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        """
            nfeat= 1433
            nhid= 8
            nclass= 7
            dropout= 0.6
            nheads= 8
            alpha= 0.2
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        
        #******
        # 两层的GAT layer
        # nfeat->nhid(nheads个) nhid*nheads->nclass
        # 穿插为k层
        #*****
        
        # nfeat--input_features 1433, nhid--7 nclass, dropout, alpha, nheadsnheads=8
        # 这里可以理解为每个self.attention是由多个GAT layer构成的，每个独立算一组权重self.W和self.a
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
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
        
        ### 这里仅仅需要的操作就剩使用不同的adj(1-->k) mask list去调用采样函数

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# In[3]:


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(mx):#D^(-1/2)^T*A*D^(-1/2) preprocess 得到一个领域传播位权矩阵
    """
    Row-normalize sparse matrix
    """
    """
    normalize_adj process
    adj--original matrix shape (2708, 2708)
    adj--degree sum matrix shape (2708, 1)
    adj--degree inv matrix shape (2708,)
    adj--degree no inf matrix shape (2708,)
    adj--degree norm diag matrix shape (2708, 2708)
    adj--res shape (2708, 2708)
    """
    ## print("normalize_adj process")
    ## print("adj--original matrix shape",mx.shape)
    rowsum = np.array(mx.sum(1)) #每行求和，求得每个节点的出度列矩阵
    ## print("adj--degree sum matrix shape",rowsum.shape)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    ## print("adj--degree inv matrix shape",r_inv_sqrt.shape)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    ## print("adj--degree no inf matrix shape",r_inv_sqrt.shape)
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    ## print("adj--degree norm diag matrix shape",r_mat_inv_sqrt.shape)
    resm = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()
    ## print("adj--res shape",resm.shape)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo() #D^(-1/2)*A*D^(-1/2)^T

def normalize(mx):#D^-(1)*A
    """Row-normalize sparse matrix"""
    """
    normalize process
    mx--original matrix shape (2708, 1433)
    mx--degree sum matrix shape (2708, 1)
    mx--degree inv matrix shape (2708,)
    mx--degree no inf matrix shape (2708,)
    mx--degree norm diag matrix shape (2708, 2708)
    mx--res shape (2708, 1433)
    """  
    ## print("normalize process")
    ## print("mx--original matrix shape",mx.shape)
    rowsum = np.array(mx.sum(1))
    ## print("mx--degree sum matrix shape",rowsum.shape)
    r_inv = np.power(rowsum, -1).flatten()
    ## print("mx--degree inv matrix shape",r_inv.shape)
    r_inv[np.isinf(r_inv)] = 0.
    ## print("mx--degree no inf matrix shape",r_inv.shape)
    r_mat_inv = sp.diags(r_inv)
    ## print("mx--degree norm diag matrix shape",r_mat_inv.shape)
    mx = r_mat_inv.dot(mx)
    ## print("mx--res shape",mx.shape)

    print("NA.shape",mx.shape)

    print("NA.dtype",mx.dtype)

    print("NA.type",type(mx))
    return mx

def laplacian(mx, norm):
    """Laplacian-normalize sparse matrix"""
    assert (all (len(row) == len(mx) for row in mx)), "Input should be a square matrix"

    return csgraph.laplacian(adj, normed = norm)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(path="Data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    """
    x.shape= (140, 1433)
    y.shape= (140, 7)
    tx.shape= (1000, 1433)
    ty.shape= (1000, 7)
    allx= (1708, 1433)
    ally= (1708, 7)
    len(graph)= 2708
    """

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        #Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum()/2))

    features = normalize(features) #将feature按照度归一化
    adj = normalize_adj(adj + sp.eye(adj.shape[0])) #将A按照两个边端点度归一化 PPR方法
    
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end+1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test


# In[4]:


import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from scipy.sparse import csgraph
import scipy.sparse.linalg
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import glob
import os


# In[5]:


adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="pubmed")


# In[6]:


fastmode="store_true"
epochs=1000
patience=100
hidden=8
dropout=0.6
nb_heads=8
alpha=0.2
lr=0.005
weight_decay=5e-4


# In[7]:


# Direct Feature Attention
model = GAT(nfeat=features.shape[1], 
                nhid=hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha)


# Linear Transformed Feature Attention
model = GAT(nfeat=features.shape[1], 
                nhid=hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha)

print("nfeat=",features.shape[1])
print("nhid=",hidden)
print("nclass=",int(labels.max())+1)
print("dropout=",dropout)
print("nheads=",nb_heads)
print("alpha=",alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)


# In[ ]:


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = epochs + 1
best_epoch = 0
for epoch in range(epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()


# In[ ]:




