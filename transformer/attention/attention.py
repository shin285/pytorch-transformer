import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        self.query_weight = nn.Linear(embedding_dim, attention_dim)
        self.key_weight = nn.Linear(embedding_dim, attention_dim)
        self.value_weight = nn.Linear(embedding_dim, attention_dim)
        self.attention_dim = attention_dim

    def forward(self, query, key, value):
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)

        # batch * max_length * attention_dim => batch * attention_dim * max_length
        transposed_key = torch.transpose(key, 1, 2)
        attention = torch.matmul(
            F.softmax(
                torch.matmul(query, transposed_key) / torch.sqrt(torch.tensor(self.attention_dim)),
                dim=-1
            ),
            value
        )
        return attention
