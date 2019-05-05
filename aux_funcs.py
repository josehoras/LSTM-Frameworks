import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle
import os


def load(input_file):
    data = open(input_file, 'r').read()     # should be simple plain text file
    chars = list(set(data))
    vocab_size = len(chars)
    ch2ix = {ch: i for i, ch in enumerate(chars)}
    ix2ch = {i: ch for i, ch in enumerate(chars)}
    return data, ch2ix, ix2ch, vocab_size


def save_model(data_vars, hyperparams, model_vars, hist_vars, t0, data_name):
    hist_vars[3] += time() - t0
    if os.path.isfile(data_name + '_model.p'):
        os.rename(data_name + '_model.p', data_name + '_model_back.p')
    pickle.dump((data_vars, hyperparams, model_vars), open(data_name + '_model.p', 'wb'))
    if os.path.isfile(data_name + '_hist.p'):
        os.rename(data_name + '_hist.p', data_name + '_hist_back.p')
    pickle.dump(hist_vars, open(data_name + '_hist.p', 'wb'))


def encode(seq, vocab_size):                            # 1-of-k encoding
    enc = np.zeros((1, vocab_size), dtype=int)
    enc[0][seq[0]] = 1
    for i in range(1, len(seq)):
        row = np.zeros((1, vocab_size), dtype=int)
        row[0][seq[i]] = 1
        enc = np.append(enc, row, axis=0)
    return enc


def python_gen(data, seq_length, char_to_idx, vocab_size, p=0):
    p = int(p)
    print(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            p = 0  # go to start of data
        x = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = encode(x, vocab_size)  # shape: (seq_length, input_dim)
        targets = encode(t, vocab_size)
        p = p + seq_length
        yield inputs, targets

def tf_gen(data, seq_length, char_to_idx, vocab_size, p=0):
    p = int(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            p = 0  # go to start of data
        a = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = np.expand_dims(encode(a, vocab_size), axis=1)  # shape: (seq_length, input_dim)
        targets = np.expand_dims(encode(t, vocab_size), axis=1)
        p = p + seq_length
        yield inputs, targets


def keras_gen(data, seq_length, char_to_idx, vocab_size, p=0):
    p = int(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            p = 0  # go to start of data
        a = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = np.expand_dims(encode(a, vocab_size), axis=0)  # shape: (1, seq_length, input_dim)
        targets = np.expand_dims(encode(t, vocab_size), axis=0)
        # print(targets.shape)
        p = p + seq_length
        yield inputs, targets


def plot(loss, smooth_loss, it, it_per_epoch, base_name=''):
    fig = plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(loss)
    plt.plot(smooth_loss)
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    plt.plot(epochs, [smooth_loss[i] for i in epochs], linestyle='', marker='o')
    print([smooth_loss[i] for i in epochs])
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylim([0, 5])
    if base_name != '':
        fig.savefig(base_name + '_plot.png')
    else:
        plt.show()
    plt.close("all")


def plot_vars(var1, var2, var3, var4, it, it_per_epoch, base_name=''):
    fig = plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(var1, label = "rnn/lstm_cell/kernel:0")
    plt.plot(var2, label = "Variable:0")
    plt.plot(var3, label = "rnn/lstm_cell/bias:0")
    plt.plot(var4, label = "Variable_1:0")
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    plt.plot(epochs, [var2[i] for i in epochs], linestyle='', marker='o')
    # print([var2[i] for i in epochs])
    # print(var1[-1])
    # print(var2[-1])
    # print(var3[-1])
    # print(var4[-1])
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.legend(loc='upper right', shadow=True, fontsize='x-small', bbox_to_anchor=(1.1, 1))
    plt.ylim([-10, 10])
    if base_name != '':
        fig.savefig(base_name + '_plot.png')
    else:
        plt.show()
    plt.close("all")