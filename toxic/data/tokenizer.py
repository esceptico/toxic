from typing import List, Optional, Sequence

import torch
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers import pre_tokenizers, decoders, trainers, Tokenizer


class SentencePieceBPETokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(
        self,
        vocab=None,
        merges=None,
        dropout: float = None,
        max_length: Optional[int] = 64
    ) -> None:
        self.tokenizer = Tokenizer(BPE(
            vocab, merges, dropout=dropout, unk_token=self.unk_token
        ))
        self.tokenizer.normalizer = BertNormalizer()  # noqa
        self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()  # noqa
        self.tokenizer.decoder = decoders.Metaspace()  # noqa
        self.tokenizer.add_special_tokens([self.pad_token, self.unk_token])

        self.tokenizer.enable_padding(pad_token=self.pad_token)
        self.tokenizer.enable_truncation(max_length)

    @classmethod
    def train(
        cls,
        dataset: Sequence[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        dropout: float = 0.0,
        max_length: Optional[int] = 64
    ) -> 'SentencePieceBPETokenizer':
        instance = cls(dropout=dropout, max_length=max_length)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[cls.pad_token, cls.unk_token]
        )
        instance.tokenizer.train_from_iterator(dataset, trainer=trainer)
        instance.tokenizer.model.dropout = 0.0
        return instance

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def serialize(self):
        return self.tokenizer.to_str()

    @classmethod
    def deserialize(cls, s: str) -> 'SentencePieceBPETokenizer':
        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_str(s)
        return tokenizer

    def encode_batch(self, batch: List[str]):
        return torch.tensor([
            e.ids for e in self.tokenizer.encode_batch(batch)
        ])
