import torch
import torch.nn as nn
from typing import Tuple

from models.bnn import BNN

class MCDropoutBNN(BNN):
    """
    Bayesian neural network using Monte Carlo Dropout.
    """
    def __init__(
            self, state_dim: int, action_dim: int, hidden_size: int = 64,
            drop_prob: float = 0.1, num_monte_carlo_samples: int = 50, device: torch.device = torch.device('cpu'),
            input_bounds: torch.Tensor = torch.Tensor([0.9, 2.6, 1.0])
    ) -> None:
        """
        Initialize Monte-Carlo Dropout Bayesian Neural Network.
        
        Args:
            hidden_size (int): Number of neurons in the hidden layer. Defaults to 64.
            drop_prob (float): Dropout probability for Monte Carlo Dropout. Defaults to 0.1.
            num_monte_carlo_samples (int): Number of monte carlo samples for the bayesian
                prediction. Defaults to 50.
        """
        super().__init__(
            state_dim=state_dim, action_dim=action_dim,
            device=device, input_bounds=input_bounds
        )
        self.model = nn.Sequential(
            # Augment input dimension for feature expansion
            nn.Linear((state_dim + action_dim) * 2, hidden_size), 
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, state_dim),
        )
        self.to(self.device)
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.name = "Monte Carlo Dropout Bayesian Neural Network"

    def enable_dropout(self, enable: bool) -> None:
        """
        Enable or disable dropout layers.

        Args:
            enable (bool): If True, enables dropout layers.
        """
        if enable:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        else:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()

    def bayesian_pred(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Bayesian prediction by running the model multiple times (MC Dropout)
        and calculate the mean and variance of the predictions.

        Args:
            states (torch.Tensor): Input state tensor (batch_size, state_dim).
            actions (torch.Tensor): Input action tensor (batch_size, state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the predictions.
        """
        self.model.train()  # Ensure Dropout is enabled
        H = states.shape[0]
        Ds = states.shape[-1]
        Da = actions.shape[-1]
        if self.device is not None:
            state = states.to(self.device)
            action = actions.to(self.device)
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