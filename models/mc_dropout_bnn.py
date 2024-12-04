import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MCDropoutBNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, drop_prob=0.5, device=None):
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

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

    def bayesian_pred(self, state, action, num_samples=50): # TODO: how many repeats do we need to get good distribution prediction?
        """
        Perform Bayesian prediction by running the model multiple times (MC Dropout)
        and calculate the mean and variance of the predictions.y

        Takes a batch of input samples (s,a), automatically repeat each for num_samples times.
        return a list of input batch size 

        Args:
            state (torch.Tensor): torch.Size([B, state_dim])
            action (torch.Tensor): torch.Size([B, action_dim])
            num_samples (int): Number of Monte Carlo samples.

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
        state_batch = state.unsqueeze(1).repeat(1,num_samples,1).view(H*num_samples,Ds)  # torch.Size([H*num_samples, state_dim])
        action_batch = action.unsqueeze(1).repeat(1,num_samples,1).view(H*num_samples,Da)  # torch.Size([H*num_samples, action_dim])
        preds = self.forward(state_batch, action_batch).view(H, num_samples, Ds)  # torch.Size([H, num_samples, Ds])

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