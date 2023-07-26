import torch
import torch.nn as nn
from torch import cuda

class RNN(nn.Module):
    """
    Basic RNN block. This represents a single layer of RNN
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int) -> None:
        """
        Args:
            input_size (int): Number of features for the input vector
            hidden_size (int): Number of hidden neurons
            num_layers (int):
            num_classes (int):
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*28, num_classes) # need to parametrize 28=sequence length

        
    def forward(self, x):
        """
        Args:
            x: Input vector
        Returns:

        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device = "cuda" if cuda.is_available() else "cpu")
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out