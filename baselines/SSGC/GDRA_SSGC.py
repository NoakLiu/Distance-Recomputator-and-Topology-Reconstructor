import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_preprocesing import load_data
from data_preprocesing import accuracy

class S2GCLayer(nn.Module):
    def __init__(self, in_features, out_features, K, dropout):
        super(S2GCLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.alpha = nn.Parameter(torch.Tensor(K))
        self.batch_norm = nn.BatchNorm1d(out_features)

        # Better initialization
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.ones_(self.alpha)

    def forward(self, x, adj):
        residual = x
        for k in range(self.K):
            x = torch.mm(adj, x) + self.alpha[k] * x

        out = torch.mm(x, self.weight)
        out = self.batch_norm(out)

        if self.in_features == self.out_features:  # Allow for skip connection
            out = out + residual

        return F.dropout(F.relu(out), self.dropout, training=self.training)


class S2GC(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K):
        super(S2GC, self).__init__()

        self.layer1 = S2GCLayer(nfeat, nhid, K, dropout)
        self.layer2 = S2GCLayer(nhid, nhid, K, dropout)  # Add another hidden layer
        self.layer3 = S2GCLayer(nhid, nclass, K, dropout)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.layer3(x, adj)
        return F.log_softmax(x, dim=1)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, x, adj):
        h = self.fc(x)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        e = self.attn(a_input)

        attention = F.softmax(e.view(N, N), dim=1)
        attention = attention * adj

        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_features)
        self.gat2 = GATLayer(hidden_features, out_features)

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, adj)
        return x


class S2GC_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, nheads, alpha):
        super(S2GC_GAT, self).__init__()
        self.gat = GAT(nfeat, nhid, nhid)  # 假设GAT输出和S2GC输入维度相同
        self.s2gc_layers = [S2GCLayer(nhid, nhid, K, dropout) for _ in range(nheads - 1)]
        self.out_layer = S2GCLayer(nhid * nheads, nclass, K, dropout)
        self.alpha = alpha

        for i, layer in enumerate(self.s2gc_layers):
            self.add_module('s2gc_layer_{}'.format(i), layer)

    def forward(self, x, adj):
        # GAT处理
        gat_output = self.gat(x, adj)

        # 调整邻接矩阵（可选的，根据需求实现）
        # adj = self.adjust_adj(adj, gat_output)

        # S2GC处理
        x = torch.cat([layer(gat_output, adj) for layer in self.s2gc_layers], dim=1)
        x = F.elu(self.out_layer(x, adj))
        return F.log_softmax(x, dim=1)

    # def adjust_adj(self, adj, node_features):
    #     # 根据node_features调整邻接矩阵的逻辑...
    #     return adj

    def adjust_adj_list(self, adj, node_scores, lambda_val=0.7):
        N = adj.shape[0]
        adj_list = [adj]

        for _ in range(1, self.K):
            adj_k = torch.spmm(adj_list[-1], adj)
            adj_list.append(adj_k)

        # score_diff = torch.abs(node_scores.unsqueeze(0) - node_scores.unsqueeze(1))
        # print(score_diff.shape)

        # 这里我希望搞一个mask化的移动标准，有边的话才计算两者之间的和并check是否超过threshold

        for k in range(self.K - 1):
            change_mask = torch.where(node_scores > lambda_val, torch.ones_like(adj), torch.zeros_like(adj))
            adj_list[k] = adj_list[k] * change_mask
            adj_list[k + 1] = adj_list[k + 1] * (1 - change_mask)

        return adj_list
