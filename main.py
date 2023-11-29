from torch import nn, optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from dataset import MultiLingualDataset
from transformer.transformer import Transformer
from vocabbuilder import vocab_builder
from tqdm import tqdm

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def load_data(split, source_language, target_language):
    with open(f'data/{split}.{source_language}', 'r') as f:
        source_sentences = [line.strip() for line in f if len(line.strip()) != 0]

    with open(f'data/{split}.{target_language}', 'r') as f:
        target_sentences = [line.strip() for line in f if len(line.strip()) != 0]

    return list(zip(source_sentences, target_sentences))


train_iter = load_data('train', 'de', 'en')
valid_iter = load_data('val', 'de', 'en')

de_vocabs, en_vocabs = vocab_builder(train_iter, de_tokenizer, en_tokenizer)

max_length = 64

train_dataset = MultiLingualDataset(train_iter, de_vocabs, de_tokenizer, en_vocabs, en_tokenizer, max_length)
valid_dataset = MultiLingualDataset(valid_iter, de_vocabs, de_tokenizer, en_vocabs, en_tokenizer, max_length)

encoder_vocab_size = len(de_vocabs)
decoder_vocab_size = len(en_vocabs)
batch_size = 8
num_head = 8
embedding_dim = 512
num_layer = 6
padding_idx = 0
transformer = Transformer(encoder_vocab_size, decoder_vocab_size, max_length, num_head, embedding_dim, num_layer,
                          padding_idx)

criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

epochs = 10

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

for epoch in range(epochs):
    transformer.train()
    losses = 0
    train_dataloader = tqdm(train_dataloader)
    for step, (src, tgt) in enumerate(train_dataloader):
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = transformer(src, tgt_input)
        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
        train_dataloader.set_description(
            f"Epoch: {epoch + 1}/{epochs}, Train loss: {losses / (step + 1):.5f}"
        )

    transformer.eval()
    losses = 0
    val_dataloader = tqdm(val_dataloader)
    for step, (src, tgt) in enumerate(val_dataloader):
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = transformer(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        losses += loss.item()
        train_dataloader.set_description(
            f"Epoch: {epoch + 1}/{epochs}, Validation loss: {losses / (step + 1):.5f}"
        )
