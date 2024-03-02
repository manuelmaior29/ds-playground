import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand(L.LightningModule):
    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)


    def lstm_unit(self, input_value, long_memory, short_memory):
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + 
                                              (input_value * self.wlr2) +
                                              self.blr1)
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + 
                                                   (input_value * self.wpr2) +
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        updated_long_memory = ((long_memory * long_remember_percent) +
                               (potential_remember_percent * potential_memory))
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                       self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        return ([updated_long_memory, updated_short_memory])

    def forward(self, input):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass