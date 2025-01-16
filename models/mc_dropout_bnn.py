import torch
import torch.nn as nn
from typing import Dict

class MCDropoutBNN(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_size=64, drop_prob=0.5,
            num_monte_carlo_samples=50, device=None, input_bound:torch.Tensor = torch.Tensor([0.9, 2.6, 1.0])
    ):
        """
        BNN Initialization

        Args:
            num_monte_carlo_samples: number of samples used for bayesian prediction
            input_bound: define absolute bounds for input states and actions, for input expansion
                         inputs should ideally stay within the boundary
                         given as [state1, state2, ..., action1, ...]
        """
        super(MCDropoutBNN, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear((state_dim + action_dim) * 2, hidden_size), # augment dim for symmetric inputs
            nn.ReLU(inplace = True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, state_dim),
        )
        if self.device is not None:
            self.to(self.device)
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.name = "Monte Carlo Dropout Bayesian Neural Network"
        self.input_bound = input_bound.to(self.device)

    def forward(self, state, action):
        """
        expand inputs by constructing symmetric value for each dim
        """
        with torch.no_grad():
            x = torch.cat([state, action], dim=1)
            x_1 = torch.tanh(2*x/self.input_bound)
            x_2 = torch.sqrt(1-torch.square(x_1))
            x_exp = torch.cat((x_1, x_2), dim=1)
        return self.model(x_exp)

    def bayesian_pred(self, state, action):
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

    def enable_dropout(self, enable):
        """
        Function to enable/disable the dropout layers during inference.
        """
        if enable:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        else:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
    
    def reset_weights(self):
        """
        Resets the weights of the network.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                # Use the default initialization
                layer.reset_parameters()
            
    def load_state_dict(self, params):
        params = {k.replace('model.', ''): v.to(self.device) for k, v in params.items()}
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