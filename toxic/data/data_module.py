import math
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

from toxic.data.dataset import Dataset, collate_fn
from toxic.data.tokenizer import SentencePieceBPETokenizer


def train_val_split(dataset, val_size: float = 0.2, random_state=0xDEAD):
    generator = torch.Generator().manual_seed(random_state)
    val_size = math.ceil(len(dataset) * val_size)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator)


class DataModule:
    def __init__(self, path_to_data, vocab_size=1000):
        self.path_to_data = Path(path_to_data)
        dataset = Dataset(self.path_to_data)
        self.train_dataset, self.val_dataset = train_val_split(dataset)
        self.tokenizer = SentencePieceBPETokenizer.train(
            [item['text'] for item in self.train_dataset],
            vocab_size=vocab_size,
            dropout=0.1
        )

    def train_loader(self, batch_size: int = 512):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn(self.tokenizer),
            batch_size=batch_size
        )

    def val_loader(self, batch_size: int = 128):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn(self.tokenizer),
            batch_size=batch_size
        )

    def to_device(self, batch, device):
        text = batch['text'].to(device)
        target = {k: v.to(device) for k, v in batch['target'].items()}
        return {'text': text, 'target': target}
