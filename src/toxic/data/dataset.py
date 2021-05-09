from typing import Any, Dict, List, Tuple, Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset

from src.toxic.data.tokenizer import SentencePieceBPETokenizer


def collate(tokenizer: SentencePieceBPETokenizer) -> Callable:
    def inner(batch):
        text, target = zip(*batch)
        text = tokenizer.encode_batch(text)
        target = torch.tensor(target, dtype=torch.float)
        return {'text': text, 'target': target}
    return inner


class Dataset(TorchDataset):
    def __init__(self, data: List[Dict[str, Any]], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, List[int]]:
        """Returns one data pair (source and target)."""
        item = self.data[index]
        text = item['text']
        if self.transform is not None:
            text = self.transform(text)
        return text, item['target']

    def loader(
        self,
        tokenizer: SentencePieceBPETokenizer,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate(tokenizer),
            num_workers=num_workers,
            shuffle=shuffle
        )

