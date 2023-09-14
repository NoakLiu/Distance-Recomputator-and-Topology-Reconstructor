# Sparse Computation Module

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    # 这个方程是一个稀疏矩阵的乘法
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    # 前向传播，调用函数进行运算
    def forward(ctx, indices, values, shape, b):  # forward return computation result
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    # 反向传播，使用梯度进行计算
    @staticmethod
    def backward(ctx, grad_output):  # backward return loss gradient
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


class spGraphKHopDecentDiffAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, hop_num, dropout, alpha, beta, concat=True):
        super(spGraphKHopDecentDiffAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.hop_num = hop_num
        self.beta = beta

        self.W = nn.Parameter(torch.zeros(size=(in_features * hop_num, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features * hop_num)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, k_edge_list):
        dv = 'cuda' if input.is_cuda else 'cpu'
        print("input:", input)
        print("input.shape:", input.shape)

        N = input.size()[0]  # N是节点数 19717
        # edge = adj.nonzero().t()   #edge: torch.Size([2, 108365])

        h_prime_return = torch.zeros(N, self.out_features)

        for i in range(self.hop_num - 1, -1, -1):

            # h = torch.mm(input, self.W)

            h = torch.mm(input, self.W[i * self.in_features:(i + 1) * self.in_features,
                                :])  # (N,feature)-->(N, output) h.shape: torch.Size([19717, 8]) h: N x out

            assert not torch.isnan(h).any()

            cur_edge = k_edge_list[i]
            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h[cur_edge[0, :], :], h[cur_edge[1, :], :]), dim=1).t()  # edge是从头到尾转移
            # h: N x out; edge: 2 x E; h[edge[0,:],:]: E x out; edge_h: E x 2*out; edge_h (t): 2*out x E
            print("edge_h:", edge_h)
            print("edge_h.shape:", edge_h.shape)

            # edge_h.shape: torch.Size([16, 108365])-->float (0,1)
            # edge: 2*D x E

            edge_e = torch.exp(-self.leakyrelu(
                self.a[:, 2 * i * self.out_features:(2 * i + 2) * self.out_features].mm(edge_h).squeeze()))
            # a: 1 x 2*out
            # edge_h: 2*out x E
            # edge_e: 1 x E -->E
            # 每个边凝缩为一个数表示，每个v->v的点对用attention计算成一个数字，是edge(adj)的矩阵的值
            # self.a 与 edge_h的关系
            print("edge_e:", edge_e)
            print("edge_e.shape:", edge_e.shape)

            assert not torch.isnan(edge_e).any()
            # edge_e: E -->float (around 1 float)

            e_rowsum = self.special_spmm(cur_edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
            # 2 x E(s)--> N x N(d); all ones: N x 1(d)
            print("e_rowsum:", e_rowsum)  # 每行求和
            print("e_rowsum.shape:", e_rowsum.shape)
            # e_rowsum: N x 1
            # e_rowsum每个点连接的边的数目

            edge_e = self.dropout(edge_e)
            print("edge_e:", edge_e)
            print("edge_e.shape", edge_e.shape)
            # edge_e: E

            h_prime = self.special_spmm(cur_edge, edge_e, torch.Size([N, N]), h)
            # 2 x E(s)--> N x N(d); h(d):N x out
            # 考虑一节拓扑结构，聚合一阶拓扑结果（是由本身的点）
            assert not torch.isnan(h_prime).any()
            # h_prime: N x out

            h_prime = h_prime.div(e_rowsum)
            # 这里考虑到平均度拓扑结构，使用h_prime/e_rowsum得到矩阵的和
            # h_prime/e_rowsum
            # h_prime: N x out
            assert not torch.isnan(h_prime).any()

            # D^-1*A(value of GAT)

            h_prime_return = (1 - self.beta) * h_prime + self.beta * h_prime_return  # h为隐藏层

            if (i == 0):
                if self.concat:
                    # if this layer is not last layer
                    res = F.elu(h_prime_return)
                    return F.elu(h_prime_return)
                else:
                    # if this layer is last layer,
                    return h_prime_return

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'