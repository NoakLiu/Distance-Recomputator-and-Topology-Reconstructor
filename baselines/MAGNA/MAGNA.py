from data_preprocesing import accuracy
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
import time
import os
import glob
import torch.nn.functional as F
from data_preprocesing import load_data, accuracy

import argparse

parser = argparse.ArgumentParser(description='Parameters for MAGNABlock model')

# Model-related parameters
#parser.add_argument('--input_dim', type=int, default=128, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
#parser.add_argument('--output_dim', type=int, default=128, help='Output dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for diffused attention')
parser.add_argument('--theta', type=float, default=0.5, help='Theta value for diffused attention')
parser.add_argument('--K', type=int, default=2, help='Diffused attention loop parameter')
parser.add_argument('--dataset', type=str, default="pubmed", choices=['cora', 'pubmed'], help='Dataset choice')
# The following parameters seem implied, but were not explicitly set in the given code:
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')  # You used this but did not define its value
parser.add_argument('--fastmode', action='store_true', default=False, help='Skip validation during training')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')  # Similarly, used but not defined
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')  # Similarly, used but not defined

args = parser.parse_args()

# The rest of your code can now use args.input_dim, args.hidden_dim, etc. to get the set values.
#input_dim = args.input_dim
hidden_dim = args.hidden_dim
#output_dim = args.output_dim
num_heads = args.num_heads
alpha = args.alpha
theta = args.theta
K = args.K
dataset = args.dataset
lr = args.lr
weight_decay = args.weight_decay
dropout = args.dropout
fastmode = args.fastmode
epochs = args.epochs
patience = args.patience


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


# # Example Usage
# input_dim = 128
# hidden_dim = 64
# output_dim = 128
# num_heads = 8
# alpha = 0.5
# theta = 0.5
# K = 2

adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=dataset)
# x = torch.rand(32, input_dim)
# model = MAGNABlock(input_dim, hidden_dim, output_dim, num_heads, alpha, theta, K)
model = MAGNABlock(in_dim=features.shape[1],
                   hidden_dim=hidden_dim,
                   out_dim=int(labels.max()) + 1,
                   num_heads=num_heads,
                   alpha=alpha,
                   theta=theta,
                   K=K
                   )
# out = model(x)
# print(out.shape)

print("nfeat=", features.shape[1])
print("nclass=", int(labels.max()) + 1)
print("dropout=", dropout)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

# Rest of your code remains largely the same ...

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
