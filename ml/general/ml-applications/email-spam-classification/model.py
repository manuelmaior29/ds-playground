import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import torch.nn.utils.rnn as rnn_utils

class EmailClassifier(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Sequential([nn.Linear(768, 1024),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(),
                                  nn.Dropout(p=0.25)])
        
    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer