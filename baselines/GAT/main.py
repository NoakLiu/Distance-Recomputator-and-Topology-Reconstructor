import argparse
import time
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_preprocesing import load_data
from GAT_sp_or_dense import GAT


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch, model, optimizer, features, adj, idx_train, labels):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate(model, features, adj, idx_val, labels):
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return loss_val.item(), acc_val.item()


def test(model, features, adj, idx_test, labels):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test.item(), acc_test.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
    parser.add_argument('--nhid', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nclass', type=int, default=7, help='Number of classes.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for leaky_relu.')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 weight decay.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA training.')
    args = parser.parse_args()

    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)
    if args.cuda:
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    model = GAT(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=args.nclass,
                dropout=args.dropout,
                alpha=args.alpha,
                nheads=args.nheads)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0
    for epoch in range(args.epochs):
        loss_train, acc_train = train(epoch, model, optimizer, features, adj, idx_train, labels)
        loss_val, acc_val = validate(model, features, adj, idx_val, labels)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), 'best_gat_model.pkl')

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(acc_val))

        if epoch > args.patience and acc_val < best_val_acc:
            print("Early stopping")
            break

    # Load the best model for testing
    model.load_state_dict(torch.load('best_gat_model.pkl'))
    loss_test, acc_test = test(model, features, adj, idx_test, labels)
    print("Test results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test))
