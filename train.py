import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, train_dataloader, test_dataloader=None,
                num_epochs=10, learning_rate=0.001, plot=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loss_save = [[], []]  # Train loss, Test loss
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for state_batch, action_batch, next_state_batch in train_bar:
            optimizer.zero_grad()
            pred_next_state = model(state_batch, action_batch)
            loss = loss_fn(pred_next_state, next_state_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * state_batch.size(0)

        average_train_loss = total_train_loss / len(train_dataloader.dataset)
        loss_save[0].append(average_train_loss)

        # Testing/Validation loop
        if test_dataloader:
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for state_batch, action_batch, next_state_batch in test_dataloader:
                    pred_next_state = model(state_batch, action_batch)
                    loss = loss_fn(pred_next_state, next_state_batch)
                    total_test_loss += loss.item() * state_batch.size(0)

            average_test_loss = total_test_loss / len(test_dataloader.dataset)
            loss_save[1].append(average_test_loss)

            # Save best model
            if average_test_loss < best_loss:
                best_loss = average_test_loss
                torch.save(model.state_dict(), "best_model.pth")

        #     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.6f}, Test Loss: {average_test_loss:.6f}")
        # else:
        #     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.6f}")

    # Plot losses
    if plot:
        plt.plot(loss_save[0], label="Train Loss")
        if test_dataloader:
            plt.plot(loss_save[1], label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

class EnvTrainer():
    def __init__(self, learned_env, num_epochs, batch_size, learning_rate):
        self.learned_env = learned_env
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train_env(self, dataset):
        train_dataloader = create_dataloader(dataset, self.batch_size)
        train_model(self.learned_env.model, train_dataloader=train_dataloader, num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate)
        self.learned_env.model.eval()