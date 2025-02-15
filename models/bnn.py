from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

class BNN(nn.Module, ABC):
    """
    Abstract base class defining the structure for implementing a Bayesian neural network.
    """
    def __init__(
            self, state_dim: int, action_dim: int, input_expansion: bool, 
            state_bounds: Dict[int, np.array], action_bounds: Dict[int, np.array],
            device: torch.device = torch.device('cpu'),
    ) -> None:
        """
        Initialize Bayesian Neural Network.

        Args:
            state_dim (int): Dimensionality of the state input.
            action_dim (int): Dimensionality of the action input.
            hidden_size (int): Number of neurons in the hidden layer. Defaults to 64.
            input_expansion (bool): Flag to enable or disable input expansion.
            state_bounds (Dict[int, np.array]): Mapping of state dimension index to their
                [min, max] bounds. This is used for input expansion.
            action_bounds (Dict[int, np.array]): Mapping of action dimension index to their
                [min, max] bounds. This is used for input expansion.
            device (torch.device): The device on which to place this model.
                Defaults to CPU
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = None
        self.input_expansion = input_expansion
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.device = device
        if self.input_expansion:
            # Initialize tensor to store absolute state bound for each state dimension
            self.input_bounds = torch.empty(self.state_dim + self.action_dim)
            for state_dim_idx in range(self.state_dim):
                self.input_bounds[state_dim_idx] = torch.tensor(
                    np.max(np.abs(self.state_bounds[state_dim_idx])), 
                    dtype=torch.float32
                )
            for action_dim_idx in range(self.action_dim):
                self.input_bounds[self.state_dim + action_dim_idx] = torch.tensor(
                    np.max(np.abs(self.action_bounds[action_dim_idx])),
                    dtype=torch.float32
                )
            self.input_bounds = self.input_bounds.to(self.device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional input expansion to enforce a constant norm of the
        input to the network.

        Args:
            states (torch.Tensor): State input of shape (batch_size, state_dim).
            actions (torch.Tensor): Action input of shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, state_dim).
        """
        input = torch.cat([states, actions], dim=1)
        if self.input_expansion:
            expanded_input = self.expand_input(input)
            output = self.model(expanded_input)
        else:
            output = self.model(input)
        return output
    
    def expand_input(self, input: torch.Tensor) -> torch.tensor:
        """
        Apply tanh-based transformation to input tensor and append its symmetric complement
        to enforce constant norm of input to the network.

        Args:
            input (torch.Tensor): The input tensor containing state and action values.

        Returns:
            torch.Tensor: The expanded input tensor with additional dimensions
                          representing the symmetric complement.
        """

        # Perform input expansion to keep magnitude of input constant
        transformed_input = torch.tanh(2 * input / self.input_bounds)
        symmetric_complement = torch.sqrt(1 - torch.square(transformed_input))
        expanded_input = torch.cat((transformed_input, symmetric_complement), dim=1)
        return expanded_input
    
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
