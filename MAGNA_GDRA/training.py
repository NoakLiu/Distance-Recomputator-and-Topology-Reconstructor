import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from data_preprocesing import load_data, accuracy
from MAGNA_GDRA import MAGNAGDRA
import os

# Arguments (in a real script these would come from argparse)
args = {
    "fastmode": False,
    "epochs": 1000,
    "patience": 100,
    "hidden": 8,
    "dropout": 0.6,
    "nb_heads": 8,
    "alpha": 0.2,
    "lr": 0.005,
    "weight_decay": 5e-4,
    "beta": 0.1,
    "num_sample": 10,
    "eta": 0.9,
    "dataset": "cora",
    "theta": 0.5,  # Assuming some value for theta
    "K": 10,       # Assuming some value for K
    "lambda_val": 1.0  # Assuming some value for lambda_val
}

# Load dataset
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args["dataset"])

# Initialize the MAGNAGDRA model
model = MAGNAGDRA(
    in_dim=features.shape[1],
    hidden_dim=args["hidden"],
    out_dim=int(labels.max()) + 1,
    num_heads=args["nb_heads"],
    alpha=args["alpha"],
    theta=args["theta"],  # These need to be added to the original class definition
    K=args["K"],
    dropout=args["dropout"],
    beta=args["beta"],
    eta=args["eta"],
    lambda_val=args["lambda_val"],
    mask=adj
)

optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(features)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item(), acc_val.item()

# Training loop
t_total = time.time()
loss_values = []
acc_values = []
bad_counter = 0
best_loss = float('inf')
best_epoch = 0

for epoch in range(args["epochs"]):
    loss, acc = train(epoch)
    loss_values.append(loss)
    acc_values.append(acc)
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

    if loss < best_loss:
        best_loss = loss
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args["patience"]:
        break

# Cleanup checkpoints
for epoch in range(args["epochs"]):
    if epoch != best_epoch:
        try:
            os.remove(f'checkpoint_epoch_{epoch}.pt')
        except OSError:
            pass

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Load the best model
model.load_state_dict(torch.load(f'checkpoint_epoch_{best_epoch}.pt'))

# Evaluate on test set
model.eval()
output = model(features)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))

import matplotlib.pyplot as plt

x = [x for x in range(0, len(loss_values))]
plt.plot(x, loss_values)
plt.plot(x, acc_values)
plt.savefig("training-01.jpg")