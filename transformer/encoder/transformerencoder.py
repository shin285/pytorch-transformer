import torch
from torch import nn

from transformer.attention.multiheadattention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_head, device):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_head, device)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_feed_forward = nn.Linear(embedding_dim, embedding_dim * 4)
        self.output_feed_forward = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_embedding):
        multi_head_attention_add_norm = self.__multi_head_attention_sublayer(input_embedding)
        return self.__feed_forward_sublayer(multi_head_attention_add_norm)

    def __multi_head_attention_sublayer(self, input_embedding):
        multi_head_attention = self.multi_head_attention(input_embedding, input_embedding, input_embedding)
        multi_head_attention = self.dropout(multi_head_attention)
        return self.layer_norm(input_embedding + multi_head_attention)

    def __feed_forward_sublayer(self, multi_head_attention_add_norm):
        feed_forward_output = self.output_feed_forward(
            torch.relu(self.input_feed_forward(multi_head_attention_add_norm))
        )
        feed_forward_output = self.dropout(feed_forward_output)
        return self.layer_norm(multi_head_attention_add_norm + feed_forward_output)
