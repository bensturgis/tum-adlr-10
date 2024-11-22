import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class MCDropoutBNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, drop_prob=0.5):
        super(MCDropoutBNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(inplace = True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

    def reset_weights(self):
        """
        Resets the weights of the network.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                # Use the default initialization
                layer.reset_parameters()

class MCDropoutBNN_Wrapper:
    def __init__(self, state_dim, action_dim, hidden_size=64, drop_prob=0.5):
        self.model = MCDropoutBNN(state_dim, action_dim, hidden_size=hidden_size, drop_prob=drop_prob)
    def bayesian_pred(self, state, action, num_samples=100):
        """
        Perform Bayesian prediction by running the model multiple times (MC Dropout)
        and calculate the mean and variance of the predictions.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.
            num_samples (int): Number of Monte Carlo samples.

        Returns:
            mean_pred (torch.Tensor): The mean of the predictions.
            var_pred (torch.Tensor): The variance of the predictions.
        """
        self.model.train()  # Ensure Dropout is enabled

        preds = []
        for _ in range(num_samples):
            preds.append(self.model(state, action))  # Run forward pass with Dropout enabled

        preds = torch.stack(preds)  # Shape: (num_samples, batch_size, state_dim)

        # Calculate mean and variance along the sampling dimension
        mean_pred = preds.mean(dim=0).detach().numpy()
        var_pred = preds.var(dim=0).detach().numpy()

        return mean_pred, var_pred
    def train_model(self, train_dataloader, test_dataloader=None, 
                    num_epochs=10, learning_rate=0.001, weight_decay=1e-2, plot=False):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        loss_save = [[], []]  # Train loss, Test loss
        best_loss = float("inf")

        for epoch in range(num_epochs):
            # Training loop
            self.model.train()
            total_train_loss = 0.0
            train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for state_batch, action_batch, next_state_batch in train_bar:
                optimizer.zero_grad()
                pred_next_state = self.model(state_batch, action_batch)
                loss = loss_fn(pred_next_state, next_state_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * state_batch.size(0)

            average_train_loss = total_train_loss / len(train_dataloader.dataset)
            loss_save[0].append(average_train_loss)

            # Testing/Validation loop
            if test_dataloader:
                self.model.eval()
                total_test_loss = 0.0
                with torch.no_grad():
                    for state_batch, action_batch, next_state_batch in test_dataloader:
                        pred_next_state = self.model(state_batch, action_batch)
                        loss = loss_fn(pred_next_state, next_state_batch)
                        total_test_loss += loss.item() * state_batch.size(0)

                average_test_loss = total_test_loss / len(test_dataloader.dataset)
                loss_save[1].append(average_test_loss)

                # Save best model
                if average_test_loss < best_loss:
                    best_loss = average_test_loss
                    torch.save(self.model.state_dict(), "best_model.pth")

        # Plot losses
        if plot:
            plt.plot(loss_save[0], label="Train Loss")
            if test_dataloader:
                plt.plot(loss_save[1], label="Test Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
    def load_state_dict(self, params):
        self.model.load_state_dict(params)