import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GDRA
from models import MAGNABlock

class MAGNAGDRA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, beta, num_sample, adj, eta, theta, K):
        super(MAGNAGDRA, self).__init__()
        self.gdra = GDRA(nfeat, nhid, nclass, dropout, alpha, nheads, beta, num_sample, adj, eta)
        self.magna_block = MAGNABlock(nhid * nheads, nhid, nclass, nheads, alpha, theta, K)
        self.dropout = dropout

    def forward(self, x):
        # Pass through GDRA model
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gdra(x)

        # The output of the GDRA model is then passed through the MAGNA block
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.magna_block(x)

        # Apply final activation function
        return F.log_softmax(x, dim=1)

# Instantiate the model
# You will need to define the variables `nfeat`, `nhid`, `nclass`, `dropout`, `alpha`, `nheads`, `beta`, `num_sample`, `adj`, `eta`, `theta`, and `K`
# model = CombinedGDRAMAGNA(nfeat, nhid, nclass, dropout, alpha, nheads, beta, num_sample, adj, eta, theta, K)
