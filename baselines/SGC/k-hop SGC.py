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
    def __init__(self, in_features, out_features, adj_list):
        """
        :param in_features: Number of input features
        :param out_features: Number of output features
        :param adj_list: List of adjacency matrices for 1-hop to K-hop
        """
        super(SGC, self).__init__()
        self.adj_list = adj_list
        self.linear = nn.Linear(in_features * len(adj_list), out_features)

    def forward(self, x):
        """
        :param x: Node feature matrix
        """
        # Aggregate features from each hop
        aggregated_features = []
        for adj in self.adj_list:
            aggregated_features.append(torch.spmm(adj, x))
        # Concatenate features from all hops
        x = torch.cat(aggregated_features, dim=1)
        x = self.linear(x)
        return x

# 计算1-hop到K-hop的邻接矩阵
K = 3  # Example hop count
adj_list = [adj]  # Start with 1-hop adjacency matrix
current_adj = adj
for _ in range(1, K):
    current_adj = torch.spmm(current_adj, adj)
    adj_list.append(current_adj)

# 模型初始化
model = SGC(F, C, adj_list)

output = model(features)
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
# model = SGC(features.shape[1], int(labels.max()) + 1, K=2)
#
# print("nfeat=", features.shape[1])
# print("nclass=", int(labels.max()) + 1)
# print("dropout=", dropout)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model = SGC(features.shape[1], int(labels.max()) + 1, adj_list)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)  # Adjusted here
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        model.eval()
        output = model(features)  # Adjusted here

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
    output = model(features)  # Adjusted here
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

# import matplotlib.pyplot as plt
# x=[x for x in range(0,len(loss_values))]
# plt.plot(x,loss_values)
# plt.plot(x,acc_values)
# plt.savefig("training-02.jpg")

import matplotlib.pyplot as plt
def draw_pic(list1, list2, name, color, x_name, y_name):
    plt.figure()
    plt.plot(list1, list2, marker='o', color=color)
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig('{}.png'.format(name))
    plt.show()

x=[x for x in range(0,len(loss_values))]

print(x)
print(loss_values)
print(acc_values)

draw_pic(x, loss_values, "K-Hop SGC Test Loss versus Epochs","red","epoches","test loss")
draw_pic(x, acc_values, "K-Hop SGC Test Accuracy versus Epochs","blue","epoches","test accuracy")


