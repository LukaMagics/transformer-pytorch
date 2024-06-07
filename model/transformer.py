import torch
from torch import nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Example Config:
        max_seq_length: 15
        num_layers: 6
        num_heads: 8
        d_ff: 2048
        d_model: 512
        drop_prob: 0.1
        batch_size: 1
        src_vocab_size(tgt_vocab_size): 36440

    Torch:
        nn.Embedding(num_embeddings, embedding_dim)
        nn.ModuleList(iterable_modules)
        nn.Linear(in_features_size, out_features_size)
        nn.Softmax(dimension_sum_to_1)
        nn.Dropout(probability)

        tril(tensor): Make the tensor as lower triangular matrix
        & : if elements on the same position are True, it returns True at the position

    embedding is always different because nn.Embedding() is initialized randomly
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers, num_heads, max_seq_length, d_model, d_ff, drop_prob):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # [36440, 512]
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # [36440, 512]
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([Encoder(num_heads, d_model, d_ff, drop_prob) for _ in range(num_layers)])
        # self.decoder_layers = nn.ModuleList([Decoder(num_heads, d_model, d_ff, dropout) for _ in range(num_layers)])
        # self.linear = nn.Linear(d_model, tgt_vocab_size)
        # self.softmax = nn.Softmax(tgt_vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def generate_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [2,1,1,15]
        return src_mask

    def generate_tgt_mask(self, tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # [2,1,15,1]
        max_seq_length = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).bool()  # [15,15]
        tgt_mask = tgt_mask & subsequent_mask  # [2,1,15,15]
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        src_embedding = self.encoder_embedding(src)
        encoder_input = self.positional_encoding(src_embedding)

        encoder_output = encoder_input
        for encoder in self.encoder_layers:
            encoder_output = encoder(encoder_output, src_mask)
            break

        return encoder_output
