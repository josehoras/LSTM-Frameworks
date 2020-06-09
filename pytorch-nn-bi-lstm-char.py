import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import aux_funcs as aux
import numpy as np
from docopt import docopt


# ----------
# CONSTANTS
# ----------
device = torch.device("cuda:0")
# device = torch.device("cpu")
n_layers = 4

class char_lstm(nn.Module):
    def __init__(self, vocab, hidden_size, dropout_rate=0.2):
        super(char_lstm, self).__init__()

        self.lstm = nn.LSTM(vocab, hidden_size, n_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input, target, verbose=False):
        if verbose==True:print("Input: ", input.size())
        output, (hn, cn) = self.lstm(input)
        if verbose==True:
            print("Output: ", output.size())
            print("  Hidden: ", hn.size(), cn.size())
        scores = self.linear(output)
        if verbose==True:print("Scores: ", scores.size())
        probs = nn.functional.softmax(scores, dim=2)
        if verbose==True:print("Probs: ", probs.size())
        logprobs = -F.log_softmax(scores, dim=2)
        if verbose==True:print("Logprobs: ", logprobs.size())
        indexes = torch.max(target[:,0], 1)[1].unsqueeze(-1).unsqueeze(-1)
        if verbose == True:print("Indexes: ", indexes.size())
        correct_logprobs = torch.gather(logprobs, 2, indexes)
        if verbose == True:print("correct_logprobs: ", correct_logprobs.size())
        loss = torch.sum(correct_logprobs) / input.size()[0]
        if verbose == True:print(loss.size())
        if verbose == True:print("-" * 40)
        return loss

    def sample(self, x, txt_length=500):
        txt = ""
        vocab_size = x.size()[0]
        # print("X: ", vocab_size)
        x.unsqueeze_(0).unsqueeze_(0)
        x = x.to(device)
        h = torch.zeros(n_layers, 1, 250).to(device)
        c = torch.zeros(n_layers, 1, 250).to(device)
        for i in range(txt_length):
            output, (h, c) = self.lstm(x, (h, c))
            # print("hidden: ", hn.size(), cn.size())
            # print("Output: ", output.size())
            scores = self.linear(output)
            # print("Scores: ", scores.size())
            probs = nn.functional.softmax(scores, dim=2)
            # print("Probs: ", probs.size())
            # print(torch.sum(probs[0][0]))
            pred = np.random.choice(range(vocab_size), p=probs.detach().cpu().numpy()[0][0])
            x = aux.encode([pred], vocab_size)
            x = torch.FloatTensor([x]).to(device)
            next_character = idx_to_char[pred]
            txt += next_character
        return txt

############
### Main ###
############
# hyperparameters
learning_rate = 1e-2
seq_length = 100
# load data
data_name = 'shakespeare'
input_file = data_name +'.txt'
data, char_to_idx, idx_to_char, vocab_size = aux.load(input_file)
print('data has %d characters, %d unique.' % (len(data), vocab_size))
data_feed = aux.python_gen(data, seq_length, char_to_idx, vocab_size)

# model dimensions (more hyperparameters)
# input_dim = vocab_size
hidden_dim = 250

model = char_lstm(vocab_size, hidden_dim)
model = model.to(device)
# print(model)
params = list(model.parameters())
# inputs = seq_length one-hot vectors of length vocab_size
inputs, targets = (next(data_feed))


inputs = torch.FloatTensor([inputs])
targets = torch.FloatTensor([targets])
print("Inputs: ", inputs.shape, inputs.dtype)

model.zero_grad()     # zeroes the gradient buffers of all parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('-' * 80)
for i in range(200000):
    inputs, targets = (next(data_feed))
    # print(inputs[1])
    # print(targets[0])
    inputs = torch.FloatTensor(inputs).to(device)
    targets = torch.LongTensor(targets).to(device)
    inputs = inputs.unsqueeze(1)
    targets = targets.unsqueeze(1)

    loss = model(inputs, targets, verbose=False)
    # loss2= model.forward2(inputs, targets, verbose=False)
    # if np.abs(loss2.item() - loss.item()) > 0.001: print("BAD")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        print('-' * 80)
        print(i, ": ", loss)
        print(model.sample(inputs[0][0]))
        print('-' * 80)
