import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Basic RNN block. This represents a single layer of RNN
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int) -> None:
        """
        Args:
            input_size (int): Number of features for the input vector
            hidden_size (int): Number of hidden neurons
            output_size (int): Number of features for the output vector
            batch_size (int): Size of a batch
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input vector
            hidden_state: Previous hidden state
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Linear output (without activation), New hidden state matrix)
        """
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tahn(x + hidden_state)
        out = self.h2o*(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Args:
            batch_size (int, optional): Batch size for a zero-initialized hidden state. Defaults to 1.

        Returns:
            torch.Tensor: A hidden state with specified batch size
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)





