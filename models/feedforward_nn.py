import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(FeedforwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)