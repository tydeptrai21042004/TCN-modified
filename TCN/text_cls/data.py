# TCN/text_cls/data.py
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS, IMDB, YelpReviewPolarity
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from functools import partial

def _iter_text(dataset_iter):
    for y, x in dataset_iter:
        yield x

def build_vocab(dataset_name, min_freq=2):
    tok = get_tokenizer("basic_english")
    if dataset_name == "ag_news":
        train_iter = AG_NEWS(split='train')
    elif dataset_name == "imdb":
        train_iter = IMDB(split='train')
    elif dataset_name == "yelp_polarity":
        train_iter = YelpReviewPolarity(split='train')
    else:
        raise ValueError("Unknown dataset")

    def _yield_tokens():
        for label, text in train_iter:
            yield tok(text)

    vocab = build_vocab_from_iterator(_yield_tokens(), specials=["<pad>", "<unk>"], min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])
    return vocab, tok

def _encode_batch(batch, vocab, tok, max_len):
    texts, labels = [], []
    for y, x in batch:
        ids = [vocab[token] for token in tok(x)]
        if max_len:
            ids = ids[:max_len]
        texts.append(torch.tensor(ids, dtype=torch.long))
        labels.append(int(y) - 1)  # torchtext labels start at 1
    lengths = [len(t) for t in texts]
    L = max(lengths) if not max_len else max_len
    padded = torch.full((len(texts), L), vocab["<pad>"], dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t
    return padded, torch.tensor(labels, dtype=torch.long)

def make_loaders(dataset_name, batch_size=64, max_len=512, root=None):
    if dataset_name == "ag_news":
        train_iter, test_iter = AG_NEWS(split=('train','test'), root=root)
        num_classes = 4
    elif dataset_name == "imdb":
        train_iter, test_iter = IMDB(split=('train','test'), root=root)
        num_classes = 2
    elif dataset_name == "yelp_polarity":
        train_iter, test_iter = YelpReviewPolarity(split=('train','test'), root=root)
        num_classes = 2
    else:
        raise ValueError("Unknown dataset")

    vocab, tok = build_vocab(dataset_name)
    collate = partial(_encode_batch, vocab=vocab, tok=tok, max_len=max_len)

    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(list(test_iter),  batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0)
    return train_loader, test_loader, vocab, num_classes
