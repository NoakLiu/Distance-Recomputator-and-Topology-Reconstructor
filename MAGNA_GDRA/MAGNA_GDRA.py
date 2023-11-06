import torch
import torch.nn.functional as F
import torch.nn as nn
from models import GraphDistRecompAttentionLayer, DiffusedAttention, MultiHeadAttention
class MAGNAGDRA(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, alpha, theta, K, dropout, beta, eta, lambda_val, mask):
        super(MAGNAGDRA, self).__init__()

        # Initialize the multi-head distance-based recompute attention layers
        self.attentions = nn.ModuleList([
            GraphDistRecompAttentionLayer(in_dim, hidden_dim // num_heads, dropout=dropout, alpha=alpha, beta=beta,
                                          eta=eta, concat=True)
            for _ in range(num_heads)
        ])

        # Diffused attention
        self.diffused_attention = DiffusedAttention(hidden_dim, alpha, theta, K)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        # Linear transformation layers
        self.map_to_hidden_dim = nn.Linear(in_dim, hidden_dim)
        self.map_to_out_dim = nn.Linear(hidden_dim, out_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.dropout = dropout
        # lambda_val for controlling the distance-based attention mechanism
        self.lambda_val = lambda_val
        self.mask = mask
        self.multihead_attention = MultiHeadAttention(hidden_dim, hidden_dim, num_heads)

    def forward(self, x):
        # Apply dropout to input features
        x = F.dropout(x, self.dropout, training=self.training)

        # Concatenate the outputs of the multi-head distance-based recompute attention layers
        x = torch.cat([att(x, self.lambda_val, self.mask) for att in self.attentions], dim=1)

        # Apply dropout again
        x = F.dropout(x, self.dropout, training=self.training)

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
