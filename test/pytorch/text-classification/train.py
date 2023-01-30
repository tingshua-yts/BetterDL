import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = iter(AG_NEWS(split='train'))
print(next(train_iter))
# tokenizer = get_tokenizer('basic_english')
# train_iter = AG_NEWS(split='train')

# def yield_tokens(data_iter):
#     for _, text in data_iter:
#         yield tokenizer(text)

# vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])