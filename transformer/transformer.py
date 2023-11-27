import torch
from torch import nn

from transformer.decoder.transformerdecoder import TransformerDecoder
from transformer.encoder.transformerencoder import TransformerEncoder
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, max_length, num_head, embedding_dim, num_layer):
        super().__init__()
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.max_length = max_length
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.num_layer = num_layer

        self.encoder_text_embedding = nn.Embedding(num_embeddings=self.encoder_vocab_size,
                                                   embedding_dim=self.embedding_dim,
                                                   padding_idx=0)
        self.decoder_text_embedding = nn.Embedding(num_embeddings=self.decoder_vocab_size,
                                                   embedding_dim=self.embedding_dim,
                                                   padding_idx=0)

        self.positional_embedding = nn.Embedding(num_embeddings=self.max_length, embedding_dim=self.embedding_dim)
        self.position_input = torch.tensor([*range(self.max_length)])

        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(self.embedding_dim, self.num_head) for _ in range(num_layer)
        ])

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoder(self.embedding_dim, self.num_head) for _ in range(num_layer)]
        )

        self.transformer_output_linear = nn.Linear(self.embedding_dim, self.decoder_vocab_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, encoder_input, decoder_input):
        encoder_output = self.__encoding(encoder_input)
        decoder_output = self.__decoding(encoder_output, decoder_input)
        output_probability = F.softmax(
            self.transformer_output_linear(decoder_output),
            dim=-1
        )
        return output_probability

    def __encoding(self, encoder_input):
        encoder_embedding = self.encoder_text_embedding(encoder_input) + self.positional_embedding(self.position_input)
        encoder_embedding = self.dropout(encoder_embedding)

        for encoder in self.encoder_layers:
            encoder_embedding = encoder(encoder_embedding)
        return encoder_embedding

    def __decoding(self, encoder_output, decoder_input):
        decoder_embedding = self.decoder_text_embedding(decoder_input) + self.positional_embedding(self.position_input)
        decoder_embedding = self.dropout(decoder_embedding)
        for decoder in self.decoder_layers:
            decoder_embedding = decoder(encoder_output, decoder_embedding)
        return decoder_embedding
