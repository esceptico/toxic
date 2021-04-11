import sys

from tqdm import tqdm
import torch
from torch import nn
from torch import optim

from torchmetrics import F1


def train(model, train_loader, val_loader, n_epoch=10, lr=0.001):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epoch):
        running_loss = 0.
        for i, batch in enumerate(tqdm(train_loader, file=sys.stdout)):
            optimizer.zero_grad()

            output = model(batch['text'].to('cuda'))
            loss = criterion(output, batch['target'].to('cuda'))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(train_loader)
        f1 = validate(model, val_loader)
        print(f'Epoch {epoch:<3} ==> F1: {f1.item():.5f}')


def validate(model, loader):
    model.eval()
    f1 = F1(num_classes=4)
    with torch.no_grad():
        for batch in loader:
            output = model(batch['text'].to('cuda')).detach().cpu()
            f1.update(output.sigmoid(), batch['target'])
    return f1.compute()
