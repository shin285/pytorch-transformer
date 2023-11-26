import torch
from torch import nn

from transformer.attention.attention import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head):
        super().__init__()
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.attention_dim = embedding_dim // num_head
        self.multi_head_attention = nn.ModuleList(
            [
                Attention(self.embedding_dim, self.attention_dim) for _ in range(self.num_head)
            ]
        )
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_embedding):
        attentions = []
        query = key = value = input_embedding
        for attention in self.multi_head_attention:
            single_attention = attention(query, key, value)
            attentions.append(single_attention)
        concatenated_attention = torch.concat(attentions, dim=-1)
        return self.output_linear(concatenated_attention)
