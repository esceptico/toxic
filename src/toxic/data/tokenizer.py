from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers import pre_tokenizers, decoders, trainers, Tokenizer


# TODO: docstring
class SentencePieceBPETokenizer:
    """Custom SentencePiece tokenizer"""
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(
        self,
        vocab: Dict[str, int] = None,
        merges: List[Tuple[str, str]] = None,
        dropout: float = None,
        max_length: Optional[int] = 64
    ) -> None:
        """Constructor

        Args:
            vocab (Dict[str, int]): A dictionary of string keys and their ids.
            merges (List[Tuple[str, str]]): A list of pairs of tokens.
            dropout (float): BPE dropout
            max_length (int, optional): The max length at which to truncate.
                Defaults to `64`.
        """
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
        instance.tokenizer.model.dropout = None
        return instance

    @property
    def vocab_size(self):
        return len(self.tokenizer.get_vocab())

    def serialize(self):
        return self.tokenizer.to_str()

    @classmethod
    def deserialize(cls, s: str) -> 'SentencePieceBPETokenizer':
        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_str(s)
        return tokenizer

    def encode(self, text: str) -> Dict[str, Any]:
        encoding = self.tokenizer.encode(text)
        outputs = {
            'ids': torch.tensor(encoding.ids),
            'mask': torch.tensor(encoding.attention_mask),
            'spans': encoding.offsets,
        }
        return outputs

    def encode_batch(self, batch: List[str]):
        encodings = self.tokenizer.encode_batch(batch)
        outputs = {
            'ids': torch.tensor([e.ids for e in encodings]),
            'mask': torch.tensor([e.attention_mask for e in encodings]),
            'spans': [e.offsets for e in encodings],
        }
        return outputs
