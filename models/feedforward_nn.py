import torch
import torch.nn as nn
from typing import Dict

class FeedforwardNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(FeedforwardNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        self.name = "Standard Feedforward Neural Network"
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model with given state and action.

        Args:
            state (torch.Tensor): The state tensor of shape (batch_size, state_dim).
            action (torch.Tensor): The action tensor of shape (batch_size, action_dim).
        
        Returns:
            torch.Tensor: The output prediction of shape (batch_size, output_dim).
        """
        x = torch.cat([state, action], dim=1)
        return self.model(x)
    
    def reset_weights(self):
        """
        Resets the weights of the network.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                # Use the default initialization
                layer.reset_parameters()

    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "architecture": [
                str(layer) for layer in self.model
            ]
        }
        return parameter_dict