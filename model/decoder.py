from torch import nn
from model.multi_head_attention import MultiHeadAttention
from model.position_wise_feed_forward import PositionWiseFeedForward


class Decoder(nn.Module):
    """
    Torch:
        nn.LayerNorm(normalized_shape)
    """

    def __init__(self, num_heads, d_model, d_ff, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(num_heads, d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(drop_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x.size() == [2, 15, 512]
        # encoder_output.size() == [2, 15, 512]
        # src_mask.size() == [2, 1, 1, 15]
        # tgt_mask.size() == [2, 1, 15, 15]
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout1(attention_output))

        attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.layer_norm2(x + self.dropout2(attention_output))

        ff_output = self.position_wise_feed_forward(x)
        x = self.layer_norm3(x + self.dropout3(ff_output))

        return x
