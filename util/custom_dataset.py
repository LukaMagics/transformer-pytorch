import json
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        src = self.tokenizer.tokenize_src(data["source"])
        tgt = self.tokenizer.tokenize_tgt(data["target"])

        return {'src': src, 'tgt': tgt}
