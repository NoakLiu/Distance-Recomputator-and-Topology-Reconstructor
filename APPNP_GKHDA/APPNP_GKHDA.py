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

from GKHDA_layer import GraphKHopDecentDiffAttentionLayer
from APPNP import APPNP_Layer

class APPNP_KHopAttention_Multi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, k, beta, hop_masks, num_appnp_layers, out_features=None):
        super(APPNP_KHopAttention_Multi, self).__init__()

        self.dropout = dropout
        self.num_appnp_layers = num_appnp_layers

        # Initial transformation layer
        self.initial_layer = nn.Linear(nfeat, nhid)

        # K-Hop Attention Mechanisms and APPNP layers for each layer
        self.khop_attentions = nn.ModuleList()
        self.appnp_layers = nn.ModuleList()

        for i in range(num_appnp_layers):
            self.khop_attentions.append(
                GraphKHopDecentDiffAttentionLayer(nhid, nhid if out_features is None else out_features, dropout, alpha,
                                                  k, beta))
            self.appnp_layers.append(APPNP_Layer(nhid if out_features is None else out_features,
                                                 nhid if out_features is None else out_features, alpha=alpha,
                                                 dropout=dropout))

        # Output layer (optional)
        self.out_layer = nn.Linear(nhid if out_features is None else out_features,
                                   nclass) if nclass is not None else None

        self.hop_masks = hop_masks

    def forward(self, x):
        x = self.initial_layer(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Loop over each layer
        for i in range(self.num_appnp_layers):
            # Use K-Hop Attention
            x = self.khop_attentions[i](x, self.hop_masks)
            x = F.dropout(x, self.dropout, training=self.training)

            # Use APPNP propagation mechanism
            x = self.appnp_layers[i](x)
            x = F.dropout(x, self.dropout, training=self.training)

        # Use Output Layer if defined
        if self.out_layer:
            x = self.out_layer(x)

        return F.log_softmax(x, dim=1) if self.out_layer else x
