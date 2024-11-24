import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MCDropoutBNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, drop_prob=0.5):
        super(MCDropoutBNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(inplace = True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

    def bayesian_pred(self, state, action, num_samples=100):
        """
        Perform Bayesian prediction by running the model multiple times (MC Dropout)
        and calculate the mean and variance of the predictions.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.
            num_samples (int): Number of Monte Carlo samples.

        Returns:
            mean_pred (torch.Tensor): The mean of the predictions.
            var_pred (torch.Tensor): The variance of the predictions.
        """
        self.model.train()  # Ensure Dropout is enabled

        preds = []
        for _ in range(num_samples):
            preds.append(self.forward(state, action))  # Run forward pass with Dropout enabled

        preds = torch.stack(preds)  # Shape: (num_samples, batch_size, state_dim)
        
        # Calculate mean and variance along the sampling dimension
        mean_pred = preds.mean(dim=0).detach().numpy()
        var_pred = preds.var(dim=0).detach().numpy()

        return mean_pred, var_pred

    def enable_dropout(self, model):
        """
        Function to enable the dropout layers during inference.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def reset_weights(self):
        """
        Resets the weights of the network.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                # Use the default initialization
                layer.reset_parameters()
            
    def load_state_dict(self, params):
        self.model.load_state_dict(params)