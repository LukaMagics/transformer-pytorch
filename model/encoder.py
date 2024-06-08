from torch import nn
from model.multi_head_attention import MultiHeadAttention
from model.position_wise_feed_forward import PositionWiseFeedForward


class Encoder(nn.Module):
    """
    Torch:
        nn.LayerNorm(normalized_shape)
    """

    def __init__(self, num_heads, d_model, d_ff, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, d_model, drop_prob)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        # x.size() == [2, 15, 512], src_mask.size() == [2, 1, 1, 15]
        attention_output = self.self_attention(x, x, x, src_mask) # [2, 15, 512]
        x = self.layer_norm1(x + self.dropout1(attention_output)) # [2, 15, 512]
        ff_output = self.position_wise_feed_forward(x) # [2, 15, 512]
        x = self.layer_norm2(x + self.dropout2(ff_output)) # [2, 15, 512]
        return x
