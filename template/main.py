import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM

import numpy as np
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)

        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    
    trn_loss /= len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)

            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    return val_loss

def main():
    dataset_path = '/workspace/ANN/days1/data/shakespeare_train.txt'
    dataset = Shakespeare(dataset_path)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=1024, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=128, sampler=val_sampler)
    
    print('RNN 모델 Train & Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CharRNN(len(dataset.chars), 1024, len(dataset.chars), dataset.char_to_idx, dataset.idx_to_char, n_layers=6).to(device)
    best_val_loss = float('inf')
    best_model_path = '/workspace/ANN/days1/model/best_char_rnn_model.pth'
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    rnn_train_losses, rnn_val_losses = [], []

    for epoch in range(epochs):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        rnn_train_losses.append(trn_loss)
        rnn_val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Check if the current validation loss is the best we have seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with val loss: {val_loss:.4f}')
            
    print('LSTM 모델 Train & Test')

    model = CharLSTM(len(dataset.chars), 1024, len(dataset.chars), dataset.char_to_idx, dataset.idx_to_char, n_layers=6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    lstm_train_losses, lstm_val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = '/workspace/ANN/days1/model/best_char_lstm_model.pth'

    for epoch in range(epochs):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        lstm_train_losses.append(trn_loss)
        lstm_val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')
        # Check if the current validation loss is the best we have seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with val loss: {val_loss:.4f}')

    plt.plot(train_losses, label='RNN Train Loss')
    plt.plot(val_losses, label='RNN Validation Loss')

    plt.plot(train_losses, label='LSTM Train Loss')
    plt.plot(val_losses, label='LSTM Validation Loss')
    plt.legend()
    plt.show()