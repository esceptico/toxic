import sys

import torch
from omegaconf import DictConfig
from torch import nn
from torch import optim
from torchmetrics import F1
from tqdm import tqdm

from src.toxic.data.data_module import DataModule
from src.toxic.modelling.model import Model


class Trainer:
    """Model trainer"""

    def __init__(self, config: DictConfig, data: DataModule, model: Model):
        """Constructor.

        Args:
            config (DictConfig): General config.
            data (DataModule): Data module.
            model (Model): Model to train.
        """
        self.config = config
        self.data = data
        self.model = model

    def train(self):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.trainer.learning_rate
        )
        try:
            for epoch in range(self.config.trainer.n_epoch):
                train_loss = self._train(criterion, optimizer)
                if self.config.trainer.val_size > 0.0:
                    f1 = self._validate(criterion)
                    print(f'Epoch {epoch:<3} ==> F1: {f1.item():.5f}\t'
                          f'Loss {train_loss:.5f}')
        except KeyboardInterrupt:
            print('Keyboard interrupt...')
        finally:
            self.save_model(self.config.trainer.save_path)

    def _train(self, criterion, optimizer):
        """Train step"""
        self.model.train()
        self.model.to('cuda')
        running_loss = 0.
        bar = tqdm(self.data.train_loader, file=sys.stdout, leave=False)
        for i, batch in enumerate(bar):
            data, target = batch['text'], batch['target']
            optimizer.zero_grad()

            output = self.model(data['ids'].to('cuda'))
            loss = criterion(output.logits, target.to('cuda'))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(self.data.train_loader)
        return loss

    def _validate(self, criterion):
        """Validation step"""
        self.model.eval()
        f1 = F1(num_classes=len(self.data.labels))
        bar = tqdm(self.data.val_loader, file=sys.stdout, leave=False)
        with torch.no_grad():
            for batch in bar:
                data, target = batch['text'], batch['target']
                data = data['ids'].to('cuda')
                output = self.model(data).logits.detach().cpu()
                f1.update(output.sigmoid(), target)
        return f1.compute()

    def save_model(self, path: str):
        data = {
            'config': self.config,
            'model': self.model.state_dict(),
            'tokenizer': self.data.tokenizer.serialize()
        }
        torch.save(data, path)
