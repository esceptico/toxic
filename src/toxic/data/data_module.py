import math
import random
from typing import Sequence

from omegaconf import DictConfig

from src.toxic.data.augmentations.transforms import (
    RemoveWhitespaceTransform,
    SwapWordsTransform,
    AppendRandomLettersTransform,
    CutOutLettersTransform
)
from src.toxic.data.augmentations.composition import Compose, OneOf
from src.toxic.data.dataset import Dataset
from src.toxic.data.tokenizer import SentencePieceBPETokenizer
from src.toxic.data.utils import read_fasttext_data


def train_val_split(data: Sequence, val_size: float = 0.2, random_state=0xDEAD):
    generator = random.Random(random_state)
    indices = list(range(len(data)))
    val_size = math.ceil(len(data) * val_size)
    val_indices = set(generator.sample(indices, val_size))
    train = [item for i, item in enumerate(data) if i not in val_indices]
    val = [item for i, item in enumerate(data) if i in val_indices]
    return train, val


transform = OneOf([
    Compose([
        SwapWordsTransform(p=0.4, swap_probability=0.2),
        AppendRandomLettersTransform(p=0.3, append_probability=0.3),
        CutOutLettersTransform(p=0.3, cutout_probability=0.3)
    ]),
    RemoveWhitespaceTransform(p=0.2)
], p=0.7)


class DataModule:
    def __init__(self, config: DictConfig):
        self.config = config
        data = read_fasttext_data(self.config.data_path, self.labels)
        train, val = train_val_split(data)
        self.train_dataset = Dataset(train, transform=transform)
        self.val_dataset = Dataset(val)
        self.tokenizer = SentencePieceBPETokenizer.train(
            [text for text, _ in self.train_dataset],
            vocab_size=self.config.vocab_size,
            dropout=self.config.bpe_dropout,
        )

    @property
    def labels(self):
        return list(self.config.labels)

    @property
    def train_loader(self):
        return self.train_dataset.loader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            tokenizer=self.tokenizer,
        )

    @property
    def val_loader(self):
        return self.val_dataset.loader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            tokenizer=self.tokenizer,
        )
