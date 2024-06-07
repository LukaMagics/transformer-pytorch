import numpy as np
import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    np.sin(angle)
    np.cos(angle)
    """

    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.position_encoding = torch.zeros(max_seq_length, d_model) # [15,512]
        self.position_encoding.requires_grad = False

        pos = torch.arange(0, max_seq_length).float().unsqueeze(1) # [15,1]
        even_i = torch.arange(0, d_model, 2).float() # [256]

        self.position_encoding[:, 0::2] = torch.sin(pos / 10000 ** (even_i / d_model)) # [15,256]
        self.position_encoding[:, 1::2] = torch.cos(pos / 10000 ** (even_i / d_model)) # [15,256]

        # 10000 ** (even_i / d_model) >>> [256]
        # pos / 10000 ** (even_i / d_model) >>> [15,1] / [256] = [15, 256]
        # torch.sin(pos / 10000 ** (even_i / d_model)) >>> [15, 256]

    def forward(self, input_embedding):
        output = input_embedding + self.position_encoding
        return output
