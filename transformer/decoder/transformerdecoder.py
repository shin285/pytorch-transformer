import torch
from torch import nn

from transformer.attention.multiheadattention import MultiHeadAttention


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_head):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim, num_head)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_head)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_feed_forward = nn.Linear(embedding_dim, embedding_dim * 4)
        self.output_feed_forward = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, encoder_output, decoder_embedding):
        masked_multi_head_attention_add_norm = self.__masked_multi_head_attention_sublayer(decoder_embedding)
        multi_head_attention_add_norm = self.__multi_head_attention_sublayer(encoder_output,
                                                                             masked_multi_head_attention_add_norm)
        return self.__feed_forward_sublayer(multi_head_attention_add_norm)

    def __masked_multi_head_attention_sublayer(self, decoder_embedding):
        masked_multi_head_attention = self.masked_multi_head_attention(decoder_embedding, decoder_embedding,
                                                                       decoder_embedding, look_ahead_mask=True)
        masked_multi_head_attention = self.dropout(masked_multi_head_attention)
        return self.layer_norm(decoder_embedding + masked_multi_head_attention)

    def __multi_head_attention_sublayer(self, encoder_output, masked_multi_head_attention_add_norm):
        multi_head_attention = self.multi_head_attention(encoder_output, encoder_output,
                                                         masked_multi_head_attention_add_norm)
        multi_head_attention = self.dropout(multi_head_attention)
        return self.layer_norm(masked_multi_head_attention_add_norm + multi_head_attention)

    def __feed_forward_sublayer(self, multi_head_attention_add_norm):
        feed_forward_output = self.output_feed_forward(
            torch.relu(self.input_feed_forward(multi_head_attention_add_norm)))
        return self.layer_norm(multi_head_attention_add_norm + feed_forward_output)
