import torch.nn as nn
import torch
import numpy as np


# input_size -> 입력되는 단어 종류의 개수 -> 임베잉이 수행되기 전에 입력되는 단어의 개수는 고정되어야 함
# hidden_size -> 임베딩 feature 차원 크기 
# output_size -> 모델의 output 단어의 종류 -> input_size와 동일
# char_to_idx -> dataset이 character를 인코더할 때 사용하는 딕셔너리
# idx_to_char -> dataset이 인코딩된 character를 character화 할 때 사용하는 딕셔너리
# n_layers -> RNN(LSTM) layer 수

# forward -> 입력되면 다음 시점 글자 예측
# generate -> 입력된 seed character와 temperature, legnth를 기반으로 문장 생성
# generate_down -> 입력된 seed character와 legnth를 기반으로 temperature의 분산을 점진적으로 낮추면서 문장 생성
    # -> 일관성이 점점 높이지나, 문장은 패턴화됨
# generate_up -> 입력된 seed character와 legnth를 기반으로 temperature의 분산을 점진적으로 높이면서 문장 생성
    # -> 문장의 의미는 점점 다양해지며 일관성은 떨어짐


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, char_to_idx, idx_to_char, n_layers=5):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, hidden)
        output = self.decoder(output.contiguous().view(-1, self.hidden_size))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

    def generate(self, seed_chars, temperature=1.0, length=100, device='cpu'):
        self.eval()
        input_chars = [self.char_to_idx[ch] for ch in seed_chars]
        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden = self.init_hidden(1).to(device)
        samples = seed_chars

        with torch.no_grad():
            for _ in range(length):
                output, hidden = self(input_tensor, hidden)
                output = output[-1] / temperature
                probabilities = torch.softmax(output, dim=0).cpu().numpy()
                char_idx = np.random.choice(len(probabilities), p=probabilities)
                char = self.idx_to_char[char_idx]
                samples += char
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
        
        return samples

    def generate_up(self, seed_chars, start_temp=0.5, end_temp=2.0, length=100, device='cpu'):
        self.eval()
        input_chars = [self.char_to_idx[ch] for ch in seed_chars]
        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden = self.init_hidden(1).to(device)
        samples = seed_chars
        temperatures = np.linspace(start_temp, end_temp, length)

        with torch.no_grad():
            for temp in temperatures:
                output, hidden = self(input_tensor, hidden)
                output = output[-1] / temp
                probabilities = torch.softmax(output, dim=0).cpu().numpy()
                char_idx = np.random.choice(len(probabilities), p=probabilities)
                char = self.idx_to_char[char_idx]
                samples += char
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
        
        return samples

    def generate_down(self, seed_chars, start_temp=2.0, end_temp=0.5, length=100, device='cpu'):
        self.eval()
        input_chars = [self.char_to_idx[ch] for ch in seed_chars]
        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden = self.init_hidden(1).to(device)
        samples = seed_chars
        temperatures = np.linspace(start_temp, end_temp, length)

        with torch.no_grad():
            for temp in temperatures:
                output, hidden = self(input_tensor, hidden)
                output = output[-1] / temp
                probabilities = torch.softmax(output, dim=0).cpu().numpy()
                char_idx = np.random.choice(len(probabilities), p=probabilities)
                char = self.idx_to_char[char_idx]
                samples += char
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
        
        return samples

    

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, char_to_idx, idx_to_char, n_layers=5):
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        encoded = self.encoder(input)
        output, hidden = self.lstm(encoded, hidden)
        output = self.decoder(output.contiguous().view(-1, self.hidden_size))
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))

    def generate(self, seed_chars, temperature=1.0, length=100, device='cpu'):
        self.eval()
        input_chars = [self.char_to_idx[ch] for ch in seed_chars]
        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden = self.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        samples = seed_chars

        with torch.no_grad():
            for _ in range(length):
                output, hidden = self(input_tensor, hidden)
                output = output[-1] / temperature
                probabilities = torch.softmax(output, dim=0).cpu().numpy()
                char_idx = np.random.choice(len(probabilities), p=probabilities)
                char = self.idx_to_char[char_idx]
                samples += char
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
        
        return samples

    def generate_up(self, seed_chars, start_temp=0.5, end_temp=2.0, length=100, device='cpu'):
        self.eval()
        input_chars = [self.char_to_idx[ch] for ch in seed_chars]
        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden = self.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        samples = seed_chars
        temperatures = np.linspace(start_temp, end_temp, length)

        with torch.no_grad():
            for temp in temperatures:
                output, hidden = self(input_tensor, hidden)
                output = output[-1] / temp
                probabilities = torch.softmax(output, dim=0).cpu().numpy()
                char_idx = np.random.choice(len(probabilities), p=probabilities)
                char = self.idx_to_char[char_idx]
                samples += char
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
        
        return samples

    def generate_down(self, seed_chars, start_temp=2.0, end_temp=0.5, length=100, device='cpu'):
        self.eval()
        input_chars = [self.char_to_idx[ch] for ch in seed_chars]
        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden = self.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        samples = seed_chars
        temperatures = np.linspace(start_temp, end_temp, length)

        with torch.no_grad():
            for temp in temperatures:
                output, hidden = self(input_tensor, hidden)
                output = output[-1] / temp
                probabilities = torch.softmax(output, dim=0).cpu().numpy()
                char_idx = np.random.choice(len(probabilities), p=probabilities)
                char = self.idx_to_char[char_idx]
                samples += char
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
        
        return samples