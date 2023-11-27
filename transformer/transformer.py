import torch
from torch import nn

from transformer.encoder.transformerencoder import TransformerEncoder


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

    def forward(self, encoder_input, decoder_input):
        encoder_embedding = self.encoder_text_embedding(encoder_input) + self.positional_embedding(self.position_input)
        for encoder in self.encoder_layers:
            encoder_embedding = encoder(encoder_embedding)
        encoder_output = encoder_embedding
        return encoder_output
