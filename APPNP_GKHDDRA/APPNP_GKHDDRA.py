import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from data_preprocesing import load_data
from data_preprocesing import accuracy

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from scipy.sparse import csgraph
import scipy.sparse.linalg
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import glob
import os

from GKHDDRA import GraphKHopDecentDiffDistRecompAttentionLayer
from APPNP import APPNP_Layer
class GKHDDRA_APPNP(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, k, beta, eta, concat=True, K=10, alpha_appnp=0.1):
        super(GKHDDRA_APPNP, self).__init__()
        self.gkhddra = GraphKHopDecentDiffDistRecompAttentionLayer(in_features, out_features, dropout, alpha, k, beta,
                                                                   eta, concat)
        self.K = K
        self.alpha_appnp = alpha_appnp

    def forward(self, h, adj, lamda):
        h_prime = self.gkhddra(h, lamda)

        # APPNP's iterative message passing
        for _ in range(self.K):
            # Aggregate neighbors
            agg_neighbors = torch.spmm(adj, h_prime)
            # Combine node's own features and aggregated neighbors
            h_prime = (1 - self.alpha_appnp) * agg_neighbors + self.alpha_appnp * h

        return h_prime
