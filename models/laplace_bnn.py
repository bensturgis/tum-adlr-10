from laplace import Laplace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

from models.bnn import BNN

class LaplaceBNN(BNN):
    """
    Bayesian neural network using Laplace approximation.
    """
    def __init__(
            self, state_dim: int, action_dim: int, input_expansion: bool,
            state_bounds: Dict[int, np.array], action_bounds: Dict[int, np.array],
            hidden_size: int = 64, device: torch.device = torch.device('cpu'),
    ) -> None:
        """
        Initialize Laplace Approximation Bayesian Neural Network.

        Args:
            hidden_size (int): Number of neurons in the hidden layer. Defaults to 64.
        """
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, device=device,
            input_expansion=input_expansion, state_bounds=state_bounds,
            action_bounds=action_bounds
        )
        self.hidden_size = hidden_size

        input_dim = self.state_dim + self.action_dim
        # Augment input dimension for feature expansion
        if self.input_expansion:
            input_dim *= 2 
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, state_dim),
        )
        self.to(self.device)
        self.name = "Laplace Approximation Bayesian Neural Network"

    def fit_posterior(self, train_loader):
        """
        Fits the posterior distribution using Laplace approximation.
        Args:
            train_loader: DataLoader providing training data.
        """
        # move data to device
        state_batch = train_loader.dataset.tensors[0].to(self.device)
        action_batch = train_loader.dataset.tensors[1].to(self.device)
        next_state_batch = train_loader.dataset.tensors[2].to(self.device)

        # Feature expansion
        input = torch.cat([state_batch, action_batch], dim=1)
        if self.input_expansion:
            input = self.expand_input(input)

        new_dataset = TensorDataset(input, next_state_batch)
        new_dataloader = DataLoader(new_dataset, batch_size=train_loader.batch_size)
        
        # create approximator and fit posterior
        self.laplace_approximation = Laplace(
            self.model, likelihood="regression", subset_of_weights="last_layer", hessian_structure="diag"
        )
        self.laplace_approximation.fit(new_dataloader)

    def bayesian_pred(self, state, action, batch_size=10000000):
        """
        Performs Bayesian prediction to obtain mean and variance.

        Args:
            state: Input state tensor.
            action: Input action tensor.
            batch_size: Size of batches for processing large inputs.

        Returns:
            mean_pred: Predictive mean.
            var_pred: Predictive variance.
        """
        if self.laplace_approximation is None:
            raise ValueError("Laplace posterior not fitted. Call fit_posterior() first.")

        if self.device is not None:
            state = state.to(self.device)
            action = action.to(self.device)
        
        input = torch.cat([state, action], dim=1)
        if self.input_expansion:
            with torch.no_grad():
                input = self.expand_input(input)
            
        mean_preds = []
        var_preds = []
        
        # Process inputs in batches
        total_samples = state.size(0)
        for i in range(0, total_samples, batch_size):
            input_batch = input[i:i+batch_size]
            mean_batch, var_batch = self.laplace_approximation(input_batch, pred_type="glm")
            mean_preds.append(mean_batch)
            var_preds.append(var_batch.diagonal(dim1=-2, dim2=-1))

        # Concatenate results from all batches
        mean_pred = torch.cat(mean_preds, dim=0).detach().cpu().numpy()
        var_pred = torch.cat(var_preds, dim=0).detach().cpu().numpy()

        return mean_pred, var_pred
    
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

    
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.

        Returns:
            Dict[str, str]: Hyperparameter dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "input_expansion": self.input_expansion,
            "state_dim": self.state_dim,
            "state_bounds": {k: str(v) for k, v in self.state_bounds.items()},
            "action_dim": self.action_dim,
            "action_bounds": {k: str(v) for k, v in self.action_bounds.items()},
            "hidden_size": self.hidden_size,
            "device": self.device,
            "architecture": [
                str(layer) for layer in self.model
            ]
        }
        return parameter_dict