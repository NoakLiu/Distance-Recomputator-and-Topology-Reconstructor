import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from data_preprocesing import load_data
from data_preprocesing import accuracy

# load dataset: cora, citeseer, pubmed
dataset = "cora"
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=dataset)

# Hyper parameters
N = len(adj)  # Number of nodes
F = len(features[0])  # Number of node features
dropout = 0.2
lr =0.005
weight_decay = 0.001
fastmode = "store_true"
epochs = 1000
patience = 100
C = labels.unique().size(0)
print("F:",F)
print("C:",C)


class SGC(nn.Module):
    def __init__(self, in_features, out_features, K=2):
        """
        :param in_features: Number of input features (number of features for each node)
        :param out_features: Number of output features (number of classes, typically)
        :param K: Number of hops (i.e., how many times we aggregate information from neighbors). Equivalent to the layers in traditional GCN.
        """
        super(SGC, self).__init__()
        self.K = K
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        :param x: Node feature matrix (N x in_features)
        :param adj: Adjacency matrix (N x N)
        """
        # Compute the filter operation (essentially multiple aggregations) on input features
        for _ in range(self.K):
            x = torch.spmm(adj, x)
        x = self.linear(x)
        return x

model = SGC(F, C, K=2)

# Forward propagate the data through the model
output = model(features, adj)
print(output.shape)  # Should be [2708, C]


model = SGC(F, C, K=2)
output = model(features, adj)
print(output.shape)  # Should be [100, 7]
print(output)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import time
import os
import glob

# adj = torch.FloatTensor(normalize_adj(adj.numpy()).toarray())

# Model Initialization
model = SGC(features.shape[1], int(labels.max()) + 1, K=2)

print("nfeat=", features.shape[1])
print("nclass=", int(labels.max()) + 1)
print("dropout=", dropout)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)  # Adjusted here
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        model.eval()
        output = model(features, adj)  # Adjusted here

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), acc_val.data.item()

def compute_test():
    model.eval()
    output = model(features, adj)  # Adjusted here
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

t_total = time.time()
loss_values = []
acc_values = []
bad_counter = 0
best = epochs + 1
best_epoch = 0
for epoch in range(epochs):
    a,b=train(epoch)
    loss_values.append(a)
    acc_values.append(b)
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

import matplotlib.pyplot as plt
x=[x for x in range(0,len(loss_values))]
plt.plot(x,loss_values)
plt.plot(x,acc_values)
plt.savefig("training-01.jpg")
