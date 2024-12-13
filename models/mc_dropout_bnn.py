import torch
import torch.nn as nn
from typing import Dict

class MCDropoutBNN(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_size=64, drop_prob=0.5,
            num_monte_carlo_samples=50, device=None
    ):
        super(MCDropoutBNN, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(inplace = True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, state_dim),
        )
        if self.device is not None:
            self.to(self.device)
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.name = "Monte Carlo Dropout Bayesian Neural Network"

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

    def bayesian_pred(self, state, action): # TODO: how many repeats do we need to get good distribution prediction?
        """
        Perform Bayesian prediction by running the model multiple times (MC Dropout)
        and calculate the mean and variance of the predictions.y

        Takes a batch of input samples (s,a), automatically repeat each for num_samples times.
        return a list of input batch size 

        Args:
            state (torch.Tensor): torch.Size([B, state_dim])
            action (torch.Tensor): torch.Size([B, action_dim])

        Returns:
            mean_pred (torch.Tensor): The mean list of the predictions.
            var_pred (torch.Tensor): The variance list of the predictions.
        """
        self.model.train()  # Ensure Dropout is enabled
        H = state.shape[0]
        Ds = state.shape[-1]
        Da = action.shape[-1]
        if self.device is not None:
            state = state.to(self.device)
            action = action.to(self.device)
        # Create a batch of states with Monte Carlo samples
        state_batch = (
            state.unsqueeze(1)
            .repeat(1, self.num_monte_carlo_samples, 1)
            .view(H * self.num_monte_carlo_samples, Ds)
        )  # torch.Size([H * num_monte_carlo_samples, state_dim])

        # Create a batch of actions with Monte Carlo samples
        action_batch = (
            action.unsqueeze(1)
            .repeat(1, self.num_monte_carlo_samples, 1)
            .view(H * self.num_monte_carlo_samples, Da)
        )  # torch.Size([H * num_monte_carlo_samples, action_dim])

        # Compute predictions using the model
        preds = (
            self.forward(state_batch, action_batch)
            .view(H, self.num_monte_carlo_samples, Ds)
        )  # torch.Size([H, num_monte_carlo_samples, Ds])

        # Calculate mean and variance along the sampling dimension
        mean_pred = preds.mean(dim=1).detach().cpu().numpy()
        var_pred = preds.var(dim=1).detach().cpu().numpy()
        

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
        params = {k: v.to(self.device) for k, v in params.items()}
        self.model.load_state_dict(params)

    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "num_monte_carlo_samples": self.num_monte_carlo_samples,
            "architecture": [
                str(layer) for layer in self.model
            ]
        }
        return parameter_dict