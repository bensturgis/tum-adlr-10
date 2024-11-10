import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(states, actions, next_states, batch_size):
    state_tensor = torch.tensor(states, dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
    
    dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for state_batch, action_batch, next_state_batch in dataloader:
            optimizer.zero_grad()
            pred_next_state = model(state_batch, action_batch)
            loss = loss_fn(pred_next_state, next_state_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state_batch.size(0)
        average_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.6f}')