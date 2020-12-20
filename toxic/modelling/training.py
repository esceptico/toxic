import torch
from torch import nn


def train(model, loader, n_epoch=10, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epoch):
        running_loss = 0.
        for i, batch in enumerate(loader):
            data, target = batch

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(loader)
        print(f'Epoch {epoch:<3} ==> Loss: {loss:.5f}')
