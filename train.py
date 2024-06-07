from util.load_config import load_config
from util.tokenizer import EngKorTokenizer
from util.custom_dataset import CustomDataset
from model.transformer import Transformer

import torch
from torch.utils.data import DataLoader

# device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = load_config()

tokenizer = EngKorTokenizer(max_len=CONFIG["model"]["max_seq_length"])
src_vocab_size, tgt_vocab_size = tokenizer.get_vocab_size()

dataset = CustomDataset(CONFIG["data"]["train_data_path"], tokenizer)

data_loader = DataLoader(dataset, batch_size=CONFIG["train"]["batch_size"])

transformer = Transformer(src_vocab_size,
                          tgt_vocab_size,
                          CONFIG["model"]["num_layers"],
                          CONFIG["model"]["num_heads"],
                          CONFIG["model"]["max_seq_length"],
                          CONFIG["model"]["d_model"],
                          CONFIG["model"]["d_ff"],
                          CONFIG["model"]["drop_prob"])

for batch in data_loader:
    src = batch["src"]
    tgt = batch["tgt"]

    encoder_output = transformer(src, tgt)

    break