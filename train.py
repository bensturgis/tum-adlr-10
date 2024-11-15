import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def create_dataloader(states, actions, next_states, batch_size):
    state_tensor = torch.tensor(states, dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
    
    dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    loss_save = [[],[]]
    for epoch in range(num_epochs):
        model.train() # train epoch
        total_loss = 0.0
        for state_batch, action_batch, next_state_batch in train_dataloader:
            optimizer.zero_grad()
            pred_next_state = model(state_batch, action_batch)
            loss = loss_fn(pred_next_state, next_state_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state_batch.size(0)
        average_loss = total_loss / len(train_dataloader.dataset)
        loss_save[0].append(average_loss)
        model.eval() # test
        total_loss = 0.0
        with torch.no_grad():
            for state_batch, action_batch, next_state_batch in test_dataloader:
                pred_next_state = model(state_batch, action_batch)
                loss = loss_fn(pred_next_state, next_state_batch)
                total_loss += loss.item() * state_batch.size(0)
            test_loss = total_loss / len(test_dataloader.dataset)
            loss_save[1].append(test_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.6f}, Test Loss: {test_loss:.6f}')
    plt.plot(loss_save[0], label='train')
    plt.plot(loss_save[1], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()