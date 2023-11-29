from collections import Counter


def vocab_builder(multi_lingual_data_iterator, de_tokenizer, en_tokenizer, min_freq=1):
    de_tokens = []
    en_tokens = []
    for de, en in multi_lingual_data_iterator:
        de_tokens.extend(de_tokenizer(de))
        en_tokens.extend(en_tokenizer(en))

    de_counter = Counter(de_tokens)
    en_counter = Counter(en_tokens)

    de_tokens = {token for token, count in de_counter.items() if count >= min_freq}
    en_tokens = {token for token, count in en_counter.items() if count >= min_freq}

    de_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + sorted(de_tokens)
    en_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + sorted(en_tokens)

    return de_tokens, en_tokens
