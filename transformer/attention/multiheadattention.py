import torch
from torch import nn

from transformer.attention.scaledotproductattention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.attention_dim = embedding_dim // num_head
        self.multi_head_attention = nn.ModuleList(
            [
                ScaleDotProductAttention(self.embedding_dim, self.attention_dim) for _ in range(self.num_head)
            ]
        )
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, look_ahead_mask=False):
        attentions = []
        for head_attention in self.multi_head_attention:
            scaled_dot_product_attention = head_attention(query, key, value, look_ahead_mask)
            attentions.append(scaled_dot_product_attention)
        concatenated_attention = torch.concat(attentions, dim=-1)
        return self.output_linear(concatenated_attention)
