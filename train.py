from util.load_config import load_config
from util.tokenizer import EngKorTokenizer
from util.custom_dataset import CustomDataset

from torch.utils.data import DataLoader

CONFIG = load_config()

tokenizer = EngKorTokenizer(max_len=CONFIG["model"]["max_len"])
dataset = CustomDataset(CONFIG["data"]["train_data_path"], tokenizer)

data_loader = DataLoader(dataset, batch_size=2)

for batch in data_loader:
    batch_src = batch["src"]
    batch_tgt = batch["tgt"]
    # print(tokenizer.decode_src(batch_src))
    # print(tokenizer.decode_tgt(batch_tgt))
    break