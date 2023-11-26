import torch
from torch import nn

from transformer.attention.multiheadattention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_head):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_head)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_feed_forward = nn.Linear(embedding_dim, embedding_dim * 4)
        self.output_feed_forward = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, input_embedding):
        attention = self.multi_head_attention(input_embedding)
        add_layer_norm = self.layer_norm(input_embedding + attention)
        feed_forward_output = self.output_feed_forward(
            torch.relu(self.input_feed_forward(add_layer_norm))
        )
        return self.layer_norm(add_layer_norm + feed_forward_output)
