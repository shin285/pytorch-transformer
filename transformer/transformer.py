from torch import nn

from transformer.encoder.transformerencoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, max_length, num_head, embedding_dim):
        super().__init__()
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.max_length = max_length
        self.num_head = num_head
        self.embedding_dim = embedding_dim

        self.encoder_text_embedding = nn.Embedding(num_embeddings=self.encoder_vocab_size,
                                                   embedding_dim=self.embedding_dim,
                                                   padding_idx=0)
        self.decoder_text_embedding = nn.Embedding(num_embeddings=self.decoder_embedding,
                                                   embedding_dim=self.embedding_dim,
                                                   padding_idx=0)

        self.positional_embedding = nn.Embedding(num_embeddings=self.max_length, embedding_dim=self.embedding_dim)

        self.position_input = [*range(self.max_length)]

        self.encoder = TransformerEncoder(self.num_head, self.embedding_dim)
        self.decoder = None

    def forward(self, encoder_input, decoder_input):
        encoder_embedding = self.encoder_text_embedding(encoder_input) + self.positional_embedding(self.position_input)
        encoder_output = self.encoder(encoder_embedding)
