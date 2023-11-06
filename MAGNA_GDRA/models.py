import torch
import torch.nn as nn
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

    def forward(self, h, lamda, mask):
        # h代表(N, node_features) W为0线性变换层 用feature去预测class Wh中代表(in_nodes, out_class)
        # Wh代表了每个节点通过self.W线性变换之后，得到的7个不同的分数，然后对于7个不同的分数进行
        # 本质计算同GCN通过一步线性变化从每个节点到output
        # h is (in_nodes, node_features)
        # **self.W** is (node_features, out_class)
        # feature capturer: self.W from feature to a specific score [h*self.W]=Wh
        # feature correlation attention capturer: self.a to capture different attention weight from each output
        # and generate a N*N table that denote the node2node attention score
        # [[self.a[fh]*Wh]+[self.a[sh]*Wh^T]]

        # global mask
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


import torch
import torch.nn as nn
import torch.nn.functional as F


class GDRA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, beta, num_sample, adj, eta):
        """Dense version of GAT."""
        """
            nfeat= 1433
            nhid= 8
            nclass= 7
            dropout= 0.6
            nheads= 8
            alpha= 0.2
        """
        super(GDRA, self).__init__()
        self.dropout = dropout
        # nfeat--input_features 1433, nhid--7 nclass, dropout, alpha, nheadsnheads=8
        # 这里可以理解为每个self.attention是由多个GAT layer构成的，每个独立算一组权重self.W和self.a
        self.attentions = [
            GraphDistRecompAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, beta=beta, eta=eta, concat=True)
            for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphDistRecompAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, beta=beta,
                                                     eta=eta, concat=False)

        self.mask = adj

        # self.k_mask_list=preprocess_khop(adj, k,  num_sample)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, 0.7, self.mask) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, 0.85, self.mask))

        return F.log_softmax(x, dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_heads)
        ])
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        attention_out = []
        for i in range(self.num_heads):
            attention = self.leaky_relu(self.attention_heads[i](x))
            attention = F.softmax(attention, dim=1)
            attention_out.append(attention)
        return torch.stack(attention_out, dim=1)


class DiffusedAttention(nn.Module):
    def __init__(self, node_dim, alpha, theta, K):
        super(DiffusedAttention, self).__init__()
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.att = nn.Parameter(torch.Tensor(node_dim, node_dim))
        nn.init.xavier_uniform_(self.att.data, gain=1.414)

    def forward(self, H):
        # print("H.shape:",H.shape)
        S = torch.matmul(H, self.att)
        A = F.softmax(S, dim=-1)
        Z = H
        for _ in range(self.K):
            Z = self.alpha * Z + (1 - self.alpha) * (A*Z)# torch.matmul(A, Z)
            # print("Z.shape:",Z.shape)
        return Z

class MAGNABlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, alpha, theta, K):
        super(MAGNABlock, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.multihead_attention = MultiHeadAttention(in_dim, hidden_dim, num_heads)
        self.diffused_attention = DiffusedAttention(hidden_dim, alpha, theta, K)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self.map_to_hidden_dim = nn.Linear(in_dim, hidden_dim)
        self.map_to_out_dim = nn.Linear(hidden_dim, out_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # print(x.shape)
        # Multi-head attention
        attention_out = self.multihead_attention(x)
        out = torch.mean(attention_out, dim=1)

        # Diffusion
        out = self.diffused_attention(out)

        # map from input to hidden
        x_jump = self.map_to_hidden_dim(x)

        # Layer normalization & residual connection after attention
        out = self.layer_norm1(x_jump + out)

        # Feed-forward network (Deep Aggregation)
        out_ff = self.feed_forward(out)

        # map from hidden to output
        out_jump = self.map_to_out_dim(out)

        # Layer normalization & residual connection after feed-forward
        out = self.layer_norm2(out_jump + out_ff)
        return out
