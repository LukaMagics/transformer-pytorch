import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, drop_prob):
        super().__init__()
        self.num_heads = num_heads # 8
        self.d_model = d_model # 512
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads # 64

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scale_dot_product_attention(self, Q, K, V, mask):
        K_T = K.transpose(2, 3) # [2, 8, 64, 15]
        attention_weights = torch.matmul(Q, K_T) / math.sqrt(self.d_k) # [2, 8, 15, 64] @ [2, 8, 64, 15] = [2, 8, 15, 15]
        if mask is not None: # [2,1,1,15] -> broadcasted as [2, 8, 15, 15]
            attention_weights = attention_weights.masked_fill(mask == 0, -10000) # [2, 8, 15, 15]
        attention_prob = torch.softmax(attention_weights, dim=-1) # [2, 8, 15, 15]
        attention_scores = torch.matmul(attention_prob, V) # [2, 8, 15, 15] @ [2, 8, 15, 64] = [2, 8, 15, 64]
        return attention_scores # [2, 8, 15, 64]

    def split_heads(self, x):
        batch_size, max_seq_length, d_model = x.size() # [2, 15, 512]
        return x.view(batch_size, max_seq_length, self.num_heads, self.d_k).transpose(1, 2) # [2, 8, 15, 64]

    def concat_heads(self, x):
        batch_size, _, max_seq_length, _ = x.size() # [2, 8, 15, 64]
        return x.transpose(1, 2).contiguous().view(batch_size, max_seq_length, self.d_model) # [2, 15, 512]

    def forward(self, Q, K, V, mask):
        Q = self.split_heads(self.W_q(Q)) # [2, 8, 15, 64]
        K = self.split_heads(self.W_k(K)) # [2, 8, 15, 64]
        V = self.split_heads(self.W_v(V)) # [2, 8, 15, 64]

        attention_output = self.scale_dot_product_attention(Q, K, V, mask) # [2, 8, 15, 64]
        attention_output_concatenated = self.concat_heads(attention_output) # [2, 15, 512]
        output = self.W_o(attention_output_concatenated) # [2, 15, 512]
        # nn.MultiHeadAttention과 output 비교해보기

        return output