import torch
import torch.nn as nn
from typing import Dict

class FeedforwardNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(FeedforwardNN, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        self.name = "Standard Feedforward Neural Network"
        
    def forward(self, state, action):
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
            "architecture": [
                str(layer) for layer in self.model
            ]
        }
        return parameter_dict