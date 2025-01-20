import torch
import torch.nn as nn
from laplace import Laplace
from torch.utils.data import DataLoader, TensorDataset

class LaplaceBNN(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_size=64,
            device=None, input_bound: torch.Tensor = torch.Tensor([0.9, 2.6, 1.0])
    ):
        """
        Bayesian Neural Network Initialization using Laplace Approximation.

        Args:
            num_monte_carlo_samples: number of samples used for posterior prediction
            input_bound: define absolute bounds for input states and actions, for input expansion
                         inputs should ideally stay within the boundary
                         given as [state1, state2, ..., action1, ...]
        """
        super(LaplaceBNN, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear((state_dim + action_dim) * 2, hidden_size),  # augment dim for symmetric inputs
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, state_dim),
        )
        if self.device is not None:
            self.to(self.device)

        self.name = "Laplace Approximation Bayesian Neural Network"
        self.input_bound = input_bound.to(self.device)
        self.laplace_approximation = None

    def forward(self, state, action):
        """
        expand inputs by constructing symmetric value for each dim
        """
        x = torch.cat([state, action], dim=1)
        x_1 = torch.tanh(2*x/self.input_bound)
        x_2 = torch.sqrt(1-torch.square(x_1))
        x_exp = torch.cat((x_1, x_2), dim=1)
        return self.model(x_exp)

    def fit_posterior(self, train_loader):
        """
        Fits the posterior distribution using Laplace approximation.
        Args:
            train_loader: DataLoader providing training data.
        """
        expanded_inputs = []
        targets = []
        for state_batch, action_batch, next_state_batch in train_loader:
            # move data to device
            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)

            # Combine state and action
            x = torch.cat([state_batch, action_batch], dim=1)
            
            # Feature expansion
            x_1 = torch.tanh(2 * x / self.input_bound)
            x_2 = torch.sqrt(1 - torch.square(x_1))
            x_exp = torch.cat((x_1, x_2), dim=1)
            
            # Append expanded inputs and targets
            expanded_inputs.append(x_exp)
            targets.append(next_state_batch)

        # Concatenate all batches and create new DataLoader
        expanded_inputs = torch.cat(expanded_inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        new_dataset = TensorDataset(expanded_inputs, targets)
        new_dataloader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=True)
        
        # create approximator and fit posterior
        self.laplace_approximation = Laplace(
            self.model, likelihood="regression", subset_of_weights="all", hessian_structure="diag"
        )
        self.laplace_approximation.fit(new_dataloader)

    def bayesian_pred(self, state, action, batch_size=500):
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
        with torch.no_grad():
            x = torch.cat([state, action], dim=1)
            x_1 = torch.tanh(2 * x / self.input_bound)
            x_2 = torch.sqrt(1 - torch.square(x_1))
            x_exp = torch.cat((x_1, x_2), dim=1)
        
        mean_preds = []
        var_preds = []
        
        # Process inputs in batches
        total_samples = state.size(0)
        for i in range(0, total_samples, batch_size):
            x_exp_batch = x_exp[i:i+batch_size]
            mean_batch, var_batch = self.laplace_approximation(x_exp_batch, pred_type="glm")
            mean_preds.append(mean_batch)
            var_preds.append(var_batch.diagonal(dim1=-2, dim2=-1))

        # Concatenate results from all batches
        mean_pred = torch.cat(mean_preds, dim=0).detach().cpu().numpy()
        var_pred = torch.cat(var_preds, dim=0).detach().cpu().numpy()

        return mean_pred, var_pred

    def reset_weights(self):
        """
        Resets the weights of the network.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def load_state_dict(self, params):
        params = {k.replace('model.', ''): v.to(self.device) for k, v in params.items()}
        self.model.load_state_dict(params)

    def params_to_dict(self) -> dict:
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
