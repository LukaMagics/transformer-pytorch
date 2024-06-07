import torch
from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        pass