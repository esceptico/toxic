import torch
from torch.utils.data.dataset import Dataset as TorchDataset


def read_dataset(path):
    data_list = []
    with open(path) as file:
        for line in file:
            sep = line.index(' ')
            text, labels = line[sep + 1:].strip(), line[:sep]
            labels = labels.split(",")
            data_list.append({
                'text': text,
                'target': {
                    'normal': int('__label__NORMAL' in labels),
                    'insult': int('__label__INSULT' in labels),
                    'threat': int('__label__THREAT' in labels),
                    'obscenity': int('__label__OBSCENITY' in labels)
                }
            })
    return data_list


class Dataset(TorchDataset):
    def __init__(self, path, transform=None):
        self.data = read_dataset(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if self.transform is not None:
            return {
                'text': self.transform(item['text']),
                'target': item['target']
            }
        return item


def collate_fn(tokenizer):
    def inner(batch):
        data = []
        target = []
        # распаковываем
        for item in batch:
            data.append(item['text'])
            target.append(list(item['target'].values()))
        # энкодим
        text = tokenizer.encode_batch(data)
        return {'text': text, 'target': torch.tensor(target, dtype=torch.float)}
    return inner
