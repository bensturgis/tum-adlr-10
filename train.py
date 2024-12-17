import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

def train_model(
    model: nn.Module, train_dataloader: DataLoader, test_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10, learning_rate: float = 0.001, weight_decay: float = 1e-4,
    plot: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Trains a given model using the provided training DataLoader and optionally 
    evaluates it using a test DataLoader. Tracks and returns the training and 
    testing losses for each epoch. Optionally plots the loss curves at the end 
    of training.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): A PyTorch DataLoader providing training samples.
                                       Each batch should yield (state_batch, action_batch, next_state_batch).
        test_dataloader (Optional[DataLoader], optional): A DataLoader for evaluation/testing.
                                                         If provided, testing is performed after each epoch.
                                                         Defaults to None.
        num_epochs (int, optional): The number of training epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.001.
        weight_decay (float, optional): Weight decay (L2 regularization) factor. Defaults to 1e-4.
        plot (bool, optional): Whether to plot training and test losses after training. Defaults to False.
        save_name (str, optional): The filename to use when saving the best model state dict.
                                   Defaults to "best_model".

    Returns:
        Tuple[List[float], List[float]]:
            A tuple containing two lists:
            - The first list contains the average training loss for each epoch.
            - The second list contains the average test loss for each epoch (empty if no test_dataloader is provided).
    """
    print("*****************************************************")
    print("Starting the training process...")
    print("*****************************************************")

    # Determine which device (CPU/GPU) the model is on
    device = next(model.parameters()).device

    # Set up the optimizer (Adam) and the loss function (MSE for regression-like tasks)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Lists to store epoch-wise training and testing losses
    # epoch_losses[0]: Training losses across epochs
    # epoch_losses[1]: Testing losses across epochs
    epoch_losses = [[], []]

    # Main training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Accumulate training loss over the epoch
        epoch_train_loss = 0.0

        # Iterate over each batch in the training DataLoader
        for state_batch, action_batch, next_state_batch in train_dataloader:
            # Move data to the appropriate device
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            next_state_batch = next_state_batch.to(device)

            # Zero out any previously computed gradients
            optimizer.zero_grad()

            # Forward pass: predict the next states from the given states and actions
            predicted_next_state = model(state_batch, action_batch)

            # Compute the loss
            batch_loss = loss_fn(predicted_next_state, next_state_batch)

            # Backpropagate the error and update the model parameters
            batch_loss.backward()
            optimizer.step()

            # Accumulate training loss (multiply by batch size to get total loss for the batch)
            epoch_train_loss += batch_loss.item() * state_batch.size(0)

        # Compute the average training loss for the epoch
        mean_train_loss = epoch_train_loss / len(train_dataloader.dataset)
        epoch_losses[0].append(mean_train_loss)

        # If a test DataLoader is provided, evaluate the model on the test set
        if test_dataloader is not None:
            model.eval()  # Set model to evaluation mode

            epoch_test_loss = 0.0
            with torch.no_grad():
                for state_batch, action_batch, next_state_batch in test_dataloader:
                    # Move test data to device
                    state_batch = state_batch.to(device)
                    action_batch = action_batch.to(device)
                    next_state_batch = next_state_batch.to(device)

                    # Forward pass on test data
                    predicted_next_state = model(state_batch, action_batch)
                    test_loss = loss_fn(predicted_next_state, next_state_batch)

                    # Accumulate test loss
                    epoch_test_loss += test_loss.item() * state_batch.size(0)

            # Compute average test loss for this epoch
            mean_test_loss = epoch_test_loss / len(test_dataloader.dataset)
            epoch_losses[1].append(mean_test_loss)

        if not test_dataloader:
            print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {mean_train_loss:.5f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {mean_train_loss:.5f} Test Loss: {mean_test_loss:.5f}")

    # After training completes, optionally plot the training (and test, if available) losses
    if plot:
        plt.figure()
        plt.plot(epoch_losses[0], label="Train Loss")
        if test_dataloader is not None:
            plt.plot(epoch_losses[1], label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss Over Epochs")
        plt.legend()
        plt.show()

    # Create a deep copy of the model's state dict
    model_weights = copy.deepcopy(model.state_dict())
    return epoch_losses[0], epoch_losses[1], model_weights