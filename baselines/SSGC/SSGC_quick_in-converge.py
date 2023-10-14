import argparse
import glob
import os
import time
import numpy as np
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

parser = argparse.ArgumentParser(description="GMLP")
parser.add_argument("--fastmode", type=str, default="store_true")
parser.add_argument("--epochs",type=int,default=1000)
parser.add_argument("--patience",type=int,default=100)
parser.add_argument("--hidden",type=int,default=8)
parser.add_argument("--dropout",type=float,default=0.6)
parser.add_argument("--nb-heads",type=int,default=8)
parser.add_argument("--alpha",type=float,default=0.2)
parser.add_argument("--lr",type=float,default=0.002)
parser.add_argument("--weight-decay",type=float,default=5e-4)
parser.add_argument("--beta",type=float,default=0.1)
parser.add_argument("--num-sample",type=int,default=10)
parser.add_argument("--eta",type=float,default=0.9)
parser.add_argument("--dataset",type=str,default="cora")
parser.add_argument("--iterations",type=int,default=10)
parser.add_argument("--model_mode",type=str,default="APPNP")
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

# Model Initialization
model = S2GC(nfeat=features.shape[1],
             nhid=args.hidden,
             nclass=int(labels.max()) + 1,
             dropout=args.dropout,
             K=3,  # Adjust K as needed
             #nheads=args.nb_heads
             )

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Training and Testing methods remain the same as your previous code

# ... [Continue with training, validation, testing, and plotting]

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

    return loss_val.data.item(), acc_val.data.item()


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
