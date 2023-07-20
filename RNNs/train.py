import model
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch import cuda

def train(model: model.RNN, data: DataLoader, epochs: int, optimizer: Optimizer, loss_fn: nn.Module):
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
        epoch_losses = list()
        for X, Y in data:
            if X.shape[0] != model.batch_size:
                continue
            hidden = model.init_zero_hidden(batch_size=model.batch_size)
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)
            model.zero_grad()
            loss = 0
            for c in range(X.shape[1]):
                out, hidden = model(X[:,c].reshape(X.shape[0], 1), hidden)
                l = loss_fn(out, Y[:, c].long())
                loss += l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            epoch_losses.append(loss.detach().item() / X.shape[1])
        train_losses[epoch] = torch.tensor(epoch_losses).mean()
        print(f"\tEpoch: {epoch+1} | Loss: {train_losses[epoch]}")
        print(generate_text(model, data.dataset))
