import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_preprocesing import load_data, accuracy  # 假设这是您的数据加载和准确率计算方法
import time
import os
import glob


epochs = 1000
patience = 100
fastmode = "store_true"

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

class SGC(nn.Module):
    def __init__(self, in_features, out_features):
        super(SGC, self).__init__()
        # self.linear = nn.Linear(in_features, out_features)
        self.linear = nn.Linear(in_features * K, out_features)

    # def forward(self, x, adj_list):
    #     for adj in adj_list:
    #         x = torch.spmm(adj, x)
    #     return self.linear(x)

    def forward(self, x, adj_list):
        print("adj_list len", len(adj_list))
        aggregated_features = []
        for adj in adj_list:
            aggregated_features.append(torch.spmm(adj, x))
        # Concatenate features from all hops
        x = torch.cat(aggregated_features, dim=1)
        x = self.linear(x)
        return x

class GDRA_SGC(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, K):
        super(GDRA_SGC, self).__init__()
        self.gat = GAT(in_features, hidden_features, 1)
        self.sgc = SGC(in_features, out_features)
        self.K = K

    def forward(self, x, adj):
        node_scores = self.gat(x, adj)
        print(node_scores.shape)
        adj_list = self.adjust_adj_list(adj, node_scores)
        print(adj_list[0].shape)
        return self.sgc(x, adj_list)

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

# 加载数据集
dataset = "cora"
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=dataset)

# 初始化模型
hidden_features = 8
out_features =  labels.max().item() + 1
K = 3
model = GDRA_SGC(features.shape[1], hidden_features, out_features, K)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

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

x=[x for x in range(0,len(loss_values))]
print(x)
print(loss_values)
print(acc_values)

import matplotlib.pyplot as plt
def draw_pic(list1, list2, name, color, x_name, y_name):
    plt.figure()
    plt.plot(list1, list2, marker='o', color=color)
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig('{}.png'.format(name))
    plt.show()

draw_pic(x, loss_values, "GDRA-K-Hop-SGC Test Loss versus Epochs","red","epoches","test loss")
draw_pic(x, acc_values, "GDRA-K-Hop-SGC Accuracy versus Epochs","blue","epoches","test accuracy")
