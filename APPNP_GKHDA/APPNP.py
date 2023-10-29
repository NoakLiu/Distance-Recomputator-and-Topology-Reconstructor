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


class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, adj, nheads, iterations, device, model_mode):
        super(APPNP, self).__init__()

        self.dropout = dropout

        # Creating multiple attention layers
        self.attentions = [
            APPNP_Layer(nfeat, nhid, alpha=alpha, adj=adj, iterations=iterations, dropout=dropout, device=device,
                        model_mode=model_mode) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Output layer
        self.out_att = APPNP_Layer(nhid * nheads, nclass, alpha=alpha, adj=adj, iterations=iterations, dropout=dropout,
                                   device=device, model_mode=model_mode)

    def forward(self, x):
        # Apply dropout and then APPNP_ty layers
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)

        # Apply dropout and then the output layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))

        return F.log_softmax(x, dim=1)

parser = argparse.ArgumentParser(description="GMLP")
parser.add_argument("--fastmode", type=str, default="store_true")
parser.add_argument("--epochs",type=int,default=1000)
parser.add_argument("--patience",type=int,default=100)
parser.add_argument("--hidden",type=int,default=8)
parser.add_argument("--dropout",type=float,default=0.6)
parser.add_argument("--nb-heads",type=int,default=8)
parser.add_argument("--alpha",type=float,default=0.2)
parser.add_argument("--lr",type=float,default=0.005)
parser.add_argument("--weight-decay",type=float,default=5e-4)
parser.add_argument("--beta",type=float,default=0.1)
parser.add_argument("--num-sample",type=int,default=10)
parser.add_argument("--eta",type=float,default=0.9)
parser.add_argument("--dataset",type=str,default="pubmed") #cora
parser.add_argument("--iterations",type=int,default=10)
parser.add_argument("--model_mode",type=str,default="sparse")
parser.add_argument("--device",type=str,default="cpu")
args=parser.parse_args()
fastmode=args.fastmode
epochs=args.epochs
patience=args.patience
hidden=args.hidden
dropout=args.dropout
nb_heads=args.nb_heads
alpha=args.alpha
lr=args.lr
weight_decay=args.weight_decay
beta=args.beta
num_sample=args.num_sample
eta=args.eta
dataset=args.dataset
iterations=args.iterations
model_mode=args.model_mode
device=args.device

adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)

# Create model and optimizer
model = APPNP(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=int(labels.max()) + 1,
              dropout=args.dropout,
              alpha=args.alpha,
              adj=adj,
              nheads=args.nb_heads,
              iterations=args.iterations,
              device=args.device,
              model_mode=args.model_mode)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features)

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
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
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
