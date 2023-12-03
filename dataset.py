from torch.utils.data import Dataset
import torch


class MultiLingualDataset(Dataset):
    def __init__(self, multi_lingual_data, de_vocab, de_tokenizer, en_vocab, en_tokenizer, max_length, device):
        super().__init__()
        self.multi_lingual_data = multi_lingual_data
        self.de_vocab = de_vocab
        self.de_tokenizer = de_tokenizer
        self.en_vocab = en_vocab
        self.en_tokenizer = en_tokenizer

        self.de_vocab_to_index = {token: idx for idx, token in enumerate(de_vocab)}
        self.en_vocab_to_index = {token: idx for idx, token in enumerate(en_vocab)}

        self.de_unk_index = self.de_vocab_to_index['<UNK>']
        self.en_unk_index = self.en_vocab_to_index['<UNK>']

        self.de_pad_index = self.de_vocab_to_index['<PAD>']
        self.en_pad_index = self.en_vocab_to_index['<PAD>']

        self.bos_index = self.de_vocab_to_index['<BOS>']
        self.eos_index = self.de_vocab_to_index['<EOS>']

        self.max_length = max_length
        self.device = device

    def __getitem__(self, item):
        de, en = self.multi_lingual_data[item]
        de_token_ids = self.get_token_ids(de, self.de_tokenizer, self.de_vocab_to_index, self.de_unk_index)
        en_token_ids = self.get_token_ids(en, self.en_tokenizer, self.en_vocab_to_index, self.en_unk_index)

        de_token_ids = self.add_bos_and_eos(de_token_ids)
        en_token_ids = self.add_bos_and_eos(en_token_ids)

        de_token_ids = self.padding(de_token_ids, self.de_pad_index, max_length=self.max_length)
        en_token_ids = self.padding(en_token_ids, self.en_pad_index, max_length=self.max_length+1)

        return torch.tensor(de_token_ids).to(self.device), torch.tensor(en_token_ids).to(self.device)

    def __len__(self):
        return len(self.multi_lingual_data)

    def get_token_ids(self, sentence, tokenizer, vocab_to_index, unk_index):
        tokens = tokenizer(sentence)
        token_ids = []
        for token in tokens:
            token_ids.append(vocab_to_index.get(token, unk_index))
        return token_ids

    def padding(self, token_ids, pad_index, max_length):
        if len(token_ids) < max_length:
            return token_ids + [pad_index] * (max_length - len(token_ids))
        else:
            return token_ids[:max_length]

    def add_bos_and_eos(self, token_ids):
        return [self.bos_index] + token_ids + [self.eos_index]
