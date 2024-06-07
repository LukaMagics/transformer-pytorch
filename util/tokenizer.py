from transformers import BertTokenizer


class EngKorTokenizer:
    def __init__(self, max_len=512):
        self.eng_tokenizer = BertTokenizer.from_pretrained("jinmang2/kpfbert")
        self.kor_tokenizer = BertTokenizer.from_pretrained("jinmang2/kpfbert")
        self.max_len = max_len

    def tokenize_src(self, text):
        encoded_dict = self.eng_tokenizer.encode_plus(text,
                                                      max_length=self.max_len,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors='pt')
        return encoded_dict["input_ids"][0]

    def tokenize_tgt(self, text):
        encoded_dict = self.eng_tokenizer.encode_plus(text,
                                                      max_length=self.max_len,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors='pt')
        return encoded_dict["input_ids"][0]

    def decode_src(self, batch_tensor):
        batch_ids = [[int(id) for id in tensor] for tensor in batch_tensor]
        batch_tokens = [self.eng_tokenizer.convert_ids_to_tokens(ids) for ids in batch_ids]
        return batch_tokens

    def decode_tgt(self, batch_tensor):
        batch_ids = [[int(id) for id in tensor] for tensor in batch_tensor]
        batch_tokens = [self.eng_tokenizer.convert_ids_to_tokens(ids) for ids in batch_ids]
        return batch_tokens

    def get_vocab_size(self):
        return self.eng_tokenizer.vocab_size, self.kor_tokenizer.vocab_size