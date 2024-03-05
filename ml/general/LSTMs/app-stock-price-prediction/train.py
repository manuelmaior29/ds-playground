import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
import lightning as L
import src.model
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from src.data_utils import fetch_data_last_n_days, split_data
from sklearn.preprocessing import MinMaxScaler

def train(model, x_train, y_train_lstm, optimiser, criterion):
    num_epochs = 1000
    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))

def main():
    model = src.model.LSTM(1, 32, 2, 1)
    data = fetch_data_last_n_days('NVDA', 'Adj Close', 365)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    x_train, y_train, x_test, y_test = split_data(data, lookback=7)

    print('x_train: ', len(x_train))

    inputs = torch.from_numpy(x_train).type(torch.Tensor)
    labels = torch.from_numpy(y_train).type(torch.Tensor)

    train(model, x_train=inputs, y_train_lstm=labels, optimiser=Adam(model.parameters(), lr=0.01), criterion=nn.MSELoss(reduction='mean'))

if __name__ == "__main__":
    main()