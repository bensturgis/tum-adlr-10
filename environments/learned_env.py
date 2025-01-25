import numpy as np
import torch
from typing import Dict, Tuple, Any, Union

from models.feedforward_nn import FeedforwardNN
from models.bnn import BNN

class LearnedEnv():
    """
    An environment with a learned dynamics model.
    
    This environment simulates dynamics useing a learned model (e.g., Bayesian 
    Neural Network or Feedforward Neural Network) to predict the next state and reward.
    """
    def __init__(self, model: Union[FeedforwardNN, BNN]):
        """
        Initialize the environment with a learned model.

        Args:
            model (Union[FeedforwardNN, BNN]): The dynamics model to be used
                for state prediction. Can be either a Bayesian Neural Network
                or a standard Feedforward Neural Network.
        """
        super().__init__()
        self.model = model

    def step(
        self, action: np.array
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform single step in the environment by applying given action
        according to the learned dynamics.
        
        Args:
            action (np.ndarray): Action applied to the system.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: Next state, reward, termination flag,
                truncation flag, and additional info.
        """
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        # Model-specific prediction logic
        info = {}
        if isinstance(self.model, BNN):
            # Bayesian model prediction: returns next state and variance
            next_state, pred_var = self.model.bayesian_pred(state_tensor, action_tensor)
            # Update state
            self.state = next_state.squeeze()
            info["var"] = pred_var.squeeze()
            reward = self.differential_entropy(pred_var.squeeze())
        elif isinstance(self.model, FeedforwardNN):
            # Deterministic model prediction: only returns next state
            next_state = self.model(state_tensor, action_tensor).squeeze(0).detach().numpy()
            # Update state
            self.state = next_state.squeeze(0).detach().numpy()
            reward = 0.0  # No reward for deterministic models
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}. "
                             f"Expected BNN or FeedforwardNN.")

        # No termination and truncation conditions defined
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, info
    
    def differential_entropy(self, pred_vars: np.ndarray) -> float:
        """
        Compute the differential entropy of a multivariate Gaussian based on variances from
        bayesian inference.
        
        Args:
            pred_vars (np.ndarray): Predicted variances from bayesian inference of shape [state_dim].
                                    
        Returns:
            float: The computed differential entropy.
        """
        
        # Extract state dimension
        state_dim = pred_vars.size

        # Constant term for Gaussian differential entropy
        const = state_dim * np.log(np.sqrt(2 * np.pi * np.e))

        # Differential entropy computation
        diff_entropy = const + 0.5 * np.sum(np.log(pred_vars))

        return diff_entropy
    
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "model": self.model.params_to_dict()
        }
        return parameter_dict