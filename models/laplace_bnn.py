from laplace import Laplace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.bnn import BNN

class LaplaceBNN(BNN):
    """
    Bayesian neural network using Laplace approximation.
    """
    def __init__(
            self, state_dim: int, action_dim: int, hidden_size: int = 64,
            device: torch.device = torch.device('cpu'),
            input_bounds: torch.Tensor = torch.Tensor([0.9, 2.6, 1.0])
    ) -> None:
        """
        Initialize Laplace Approximation Bayesian Neural Network.

        Args:
            hidden_size (int): Number of neurons in the hidden layer. Defaults to 64.
        """
        super().__init__(
            state_dim=state_dim, action_dim=action_dim,
            device=device, input_bounds=input_bounds
        )
        self.model = nn.Sequential(
            # Augment input dimension for feature expansion
            nn.Linear((self.state_dim + self.action_dim) * 2, hidden_size), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, state_dim),
        )
        self.to(self.device)
        self.laplace_approximation = None
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
        x = torch.cat([state_batch, action_batch], dim=1)
        x_1 = torch.tanh(2 * x / self.input_bounds)
        x_2 = torch.sqrt(1 - torch.square(x_1))
        x_exp = torch.cat((x_1, x_2), dim=1)

        new_dataset = TensorDataset(x_exp, next_state_batch)
        new_dataloader = DataLoader(new_dataset, batch_size=train_loader.batch_size)
        
        # create approximator and fit posterior
        self.laplace_approximation = Laplace(
            self.model, likelihood="regression", subset_of_weights="last_layer", hessian_structure="diag"
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
            x_1 = torch.tanh(2 * x / self.input_bounds)
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