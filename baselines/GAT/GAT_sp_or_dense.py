import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, model_mode="sparse"):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.model_mode = model_mode  # specify model_mode: "dense" or "sparse"

        # Define the learnable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        if self.model_mode == "dense":
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            h_prime = torch.matmul(attention, h)

            if self.concat:
                return F.elu(h_prime)
            else:
                return h_prime

        # elif self.model_mode == "sparse":
        #     adj = adj.to_sparse()
        #     edges = adj._indices().t()
        #     a_input = torch.cat((h[edges[:, 0], :], h[edges[:, 1], :]), dim=1)
        #     e = self.leakyrelu(torch.mm(a_input, self.a).squeeze(1))
        #
        #     attention = -9e15 * torch.ones(N, N, device=input.device)
        #     attention[edges[:, 0], edges[:, 1]] = e
        #
        #     attention = F.softmax(attention, dim=1)
        #     attention = F.dropout(attention, self.dropout, training=self.training)
        #
        #     attention = torch.sparse_coo_tensor(adj._indices(), attention[adj._indices()[0], adj._indices()[1]],
        #                                          adj.shape)
        #     h_prime = torch.spmm(attention, h)
        #
        #     if self.concat:
        #         return F.elu(h_prime)
        #     else:
        #         return h_prime

        elif self.model_mode == "sparse":
            adj = adj.to_sparse()
            edges = adj._indices().t()
            a_input = torch.cat((h[edges[:, 0], :], h[edges[:, 1], :]), dim=1)
            e = self.leakyrelu(torch.mm(a_input, self.a).squeeze(1))

            # Apply the mask to the attention coefficients
            e = e.masked_fill(adj._values() == 0, float('-inf'))

            # Now we perform softmax on e. Since we're dealing with potentially very large graphs,
            # we'll handle this in chunks so as not to run out of memory.
            e = torch.scatter_add(e.exp(), 0, edges[:, 0], e)

            # Apply dropout
            attention = F.dropout(e, p=self.dropout, training=self.training)

            # Now, use the attention weights to weigh the node features
            h_prime = torch.zeros_like(h).scatter_add(0, edges[:, 1].unsqueeze(1).repeat(1, h.shape[1]),
                                                      attention.unsqueeze(1) * h[edges[:, 0]])

            if self.concat:
                return F.elu(h_prime)
            else:
                return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, model_mode="sparse"):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, model_mode=model_mode) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, model_mode=model_mode)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
