from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple

class BNN(nn.Module, ABC):
    """
    Abstract base class defining the structure for implementing a Bayesian neural network.
    """
    def __init__(
            self, state_dim: int, action_dim: int,
            device: torch.device = torch.device('cpu'),
            input_bounds: torch.Tensor = torch.Tensor([0.9, 2.6, 1.0])
    ) -> None:
        """
        Initialize Bayesian Neural Network.

        Args:
            state_dim (int): Dimensionality of the state input.
            action_dim (int): Dimensionality of the action input.
            hidden_size (int): Number of neurons in the hidden layer. Defaults to 64.
            device (torch.device): The device on which to place this model.
                Defaults to CPU
            input_bounds (torch.Tensor): Tensor representing bounds for states/actions.
                This is used for input feature expansion.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.input_bounds = input_bounds.to(self.device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input expansion to enforce a constant norm of the input to the
        network.

        Args:
            states (torch.Tensor): State input of shape (batch_size, state_dim).
            actions (torch.Tensor): Action input of shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, state_dim).
        """
        input_tensor = torch.cat([states, actions], dim=1)
        transformed_input = torch.tanh(2 * input_tensor / self.input_bounds)
        symmetric_complement = torch.sqrt(1 - torch.square(transformed_input))
        expanded_input = torch.cat((transformed_input, symmetric_complement), dim=1)
        return self.model(expanded_input)
    
    @abstractmethod
    def bayesian_pred(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method for Bayesian prediction.

        Args:
            states (torch.Tensor): Input state tensor (batch_size, state_dim).
            actions (torch.Tensor): Input action tensor (batch_size, state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the predictions.
        """
        pass

    def reset_weights(self) -> None:
        """
        Resets the weights of the network.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def load_state_dict(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Load model parameters.

        Args:
            params (Dict[str, torch.Tensor]): Model parameters.
        """
        params = {k.replace('model.', ''): v.to(self.device) for k, v in params.items()}
        self.model.load_state_dict(params)

    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.

        Returns:
            Dict[str, str]: Hyperparameter dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "architecture": [
                str(layer) for layer in self.model
            ]
        }
        return parameter_dict
