"""
Reference: https://github.com/Diego999/pyGAT/blob/master/layers.py
"""

from data_preprocesing import load_data
from data_preprocesing import accuracy
import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os

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
parser.add_argument("--dataset",type=str,default="cora")
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

class GraphDistRecompAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, beta, eta, concat=True):
        super(GraphDistRecompAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 
        self.beta = beta
        self.eta=eta
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        self.disc=nn.Parameter(torch.empty(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, lamda):
        # h代表(N, node_features) W为0线性变换层 用feature去预测class Wh中代表(in_nodes, out_class)
        # Wh代表了每个节点通过self.W线性变换之后，得到的7个不同的分数，然后对于7个不同的分数进行
        # 本质计算同GCN通过一步线性变化从每个节点到output
        # h is (in_nodes, node_features)
        # **self.W** is (node_features, out_class)
        # feature capturer: self.W from feature to a specific score [h*self.W]=Wh
        # feature correlation attention capturer: self.a to capture different attention weight from each output 
        # and generate a N*N table that denote the node2node attention score
        # [[self.a[fh]*Wh]+[self.a[sh]*Wh^T]]
        
        global mask
        dischange=torch.zeros(h.shape[0],h.shape[0])
        Wh=torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        ones_vec = torch.ones(h.shape[0],h.shape[0])
        zeros_vec = torch.zeros(h.shape[0],h.shape[0])
        attention = torch.where(mask>0, e, zero_vec)
        mask = torch.where(dischange > lamda, ones_vec, mask) #***替代方案是mask->cur_mask 每次更改某一些节点
        mask = torch.where(dischange < -lamda, zeros_vec, mask)
        attention = torch.where(mask>0, e, zero_vec) #只要mask大于0，就标记了
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        dischange = self._compute_distance_change(Wh)
        dischange = dischange/(max(abs(dischange.min()),dischange.max()))
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
        return dischange#self.leakyrelu(e)
    
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
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,beta, num_sample, adj, eta):
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
        self.attentions = [GraphDistRecompAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha,beta=beta,eta=eta,concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphDistRecompAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,beta=beta,eta=eta,concat=False)
        
        #self.k_mask_list=preprocess_khop(adj, k,  num_sample)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,0.7) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x,0.85))
        
        return F.log_softmax(x, dim=1)


# load dataset: cora, citeseer, pubmed
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=dataset) 

mask = adj
print(mask)

model = GDRA(nfeat=features.shape[1], 
                nhid=hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha,
                beta=beta,
                num_sample=num_sample,
                adj=adj,
                eta=eta
               )

print("nfeat=",features.shape[1])
print("nhid=",hidden)
print("nclass=",int(labels.max())+1)
print("dropout=",dropout)
print("nheads=",nb_heads)
print("alpha=",alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)


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