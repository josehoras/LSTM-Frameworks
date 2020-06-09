import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import Adam


class char_lstm(nn.Module):
    def __init__(self, vocab, hidden_size, n_layers=1):
        super(char_lstm, self).__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab
        self.lstm = nn.LSTM(vocab, hidden_size, n_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab, bias=True)

    def forward(self, input, h0=None, c0=None):
        if h0==None or c0==None:
            output, (hn, cn) = self.lstm(input)
        else:
            output, (hn, cn) = self.lstm(input, (h0, c0))
        scores = self.linear(output)
        return scores, hn, cn

    def sample(self, x, txt_length=500):
        x = x.view(1, 1, self.vocab_size)
        h = torch.zeros(self.n_layers, 1, hidden_dim).to(device)
        c = torch.zeros(self.n_layers, 1, hidden_dim).to(device)
        txt = ""
        for i in range(txt_length):
            scores, h, c = self.forward(x, h, c)
            probs = nn.functional.softmax(scores, dim=2).view(self.vocab_size)
            pred = torch.tensor(list(WeightedRandomSampler(probs, 1, replacement=True)))
            x = F.one_hot(pred, num_classes=self.vocab_size)
            x = x.view(1, 1, self.vocab_size).type(torch.FloatTensor).to(device)
            next_character = idx_to_char[pred.item()]
            txt += next_character
        return txt


class CustomDataset(Dataset):
    def __init__(self, data_name):
        self.data = open(data_name + '.txt', 'r').read()
        chars = sorted(set(self.data))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        print('data has %d characters, %d unique.' % (len(self.data), self.vocab_size))

    def __getitem__(self, index):
        x = self.char_to_idx[self.data[index]]
        x = torch.tensor([x])
        x = F.one_hot(x, num_classes=self.vocab_size)
        x = x.type(torch.FloatTensor)
        t = self.char_to_idx[self.data[index + (index < (self.__len__() - 1))]]
        t = torch.tensor([t])
        return (x.to(device), t.to(device))

    def __len__(self):
        return len(self.data)

    def params(self):
        return self.vocab_size, self.char_to_idx, self.idx_to_char


############
### Main ###
############
device = torch.device("cuda:0")
# hyperparameters
seq_length = 100
hidden_dim = 250
n_layers = 1
lr = 0.01

# Create data loader
train_data = CustomDataset('shakespeare')
train_loader = DataLoader(dataset=train_data, batch_size=seq_length, shuffle=False)

# Get important parameters from our dataset
vocab_size, char_to_idx, idx_to_char = train_data.params()

# Create our model and send it to device
model = char_lstm(vocab_size, hidden_dim, n_layers=n_layers).to(device)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Create optimizer
optimizer = Adam(model.parameters(), lr=lr)

# Initialize initial hidden and cell state
h = torch.zeros(n_layers, 1, hidden_dim).to(device)
c = torch.zeros(n_layers, 1, hidden_dim).to(device)

### TRAIN LOOP
i = 0
for inputs, targets in train_loader:

    # Forward run the model and get predictions
    scores, h, c = model(inputs, (h, c))

    loss = loss_fn(scores.squeeze(dim=1), targets.squeeze(dim=1))

    # Backpropagate the loss and update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Print loss and sample text every 100 steps
    if i % 500 == 0:
        print('-' * 80)
        print(i, ": ", loss)
        print(model.sample(inputs[0]))
        print('-' * 80)
    i += 1
print("# of batches: ", i)