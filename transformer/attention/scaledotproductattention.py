import torch
from torch import nn
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim, device):
        super().__init__()
        self.query_weight = nn.Linear(embedding_dim, attention_dim)
        self.key_weight = nn.Linear(embedding_dim, attention_dim)
        self.value_weight = nn.Linear(embedding_dim, attention_dim)
        self.attention_dim = attention_dim
        self.device = device

    def forward(self, query, key, value, padding_mask=None, look_ahead_mask=False):
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)

        # batch * max_length * attention_dim => batch * attention_dim * max_length
        transposed_key = torch.transpose(key, -1, -2)
        qk = torch.matmul(query, transposed_key) / torch.sqrt(torch.tensor(self.attention_dim))

        if padding_mask is not None:
            qk = qk.masked_fill(padding_mask, float('-inf'))

        if look_ahead_mask:
            qk = self.__masking(qk)
        qk = F.softmax(qk, dim=-1)
        attention = torch.matmul(qk, value)
        return attention

    def __masking(self, qk):
        seq_length = qk.size(-1)
        look_ahead_mask = (torch.tril(torch.ones(seq_length, seq_length)))
        look_ahead_mask.masked_fill_(look_ahead_mask == 0, float('-inf'))
        look_ahead_mask.masked_fill_(look_ahead_mask == 1, 0)
        return qk + look_ahead_mask.to(self.device)
