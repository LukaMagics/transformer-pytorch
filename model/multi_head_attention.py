import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, drop_prob):
        super().__init__()
        pass