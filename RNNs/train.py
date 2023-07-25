import model
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch import cuda
import tqdm

def train(model: model.RNN, dataloader: DataLoader, epochs: int, optimizer: Optimizer, loss_fn: nn.Module):
    """
    Train the model for a specified numer of epochs

    Args:
        model (model.RNN): RNN model to train
        data (DataLoader): Iterable Dataloader
        epochs (int): Number of epochs used for training the RNN model
        optimizer (Optimizer): Optimizer to use for each training epoch
        loss_fn (nn.Module): Function used for calculating the loss
    """
    train_losses = {}
    device = "cuda:0" if cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    print("[] Training started.")
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device).squeeze(1)
            targets = targets.to(device)

            scores = model(data)
            loss = loss_fn(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            print(loss)