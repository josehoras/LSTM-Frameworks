import numpy as np
from time import time
import pickle
import matplotlib.pyplot as plt
import tkinter as tk
from lstm_layers import *


def load(input_file):
    data = open(input_file, 'r').read()     # should be simple plain text file
    chars = list(set(data))
    vocab_size = len(chars)
    ch2ix = {ch: i for i, ch in enumerate(chars)}
    ix2ch = {i: ch for i, ch in enumerate(chars)}
    return data, ch2ix, ix2ch, vocab_size


class TinkerOutput(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RNNs")
        bottom_frame = tk.Frame(self,width=100,height=100)
        bottom_frame.pack(side=tk.BOTTOM, expand=False, fill=tk.BOTH)
        top_frame = tk.Frame(self,width=100,height=100)
        top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        vbar = tk.Scrollbar(top_frame,orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.txt = tk.Text(top_frame, bg="white smoke", height=16, width=80, wrap=tk.NONE,
                                 font=('monospace', 14), yscrollcommand=vbar.set,
                                 spacing1=2, padx=5, pady=10, borderwidth=4)
        self.txt.pack(expand=True, fill="both")
        vbar.config(command=self.txt.yview)
        bt = tk.Button(bottom_frame, text='Quit', font=10, command=self.destroy)
        bt.pack(side=tk.RIGHT, padx=10, pady=5)


def print_tk_txt(txt, cell_states):
    root = TinkerOutput()
    # define color tags
    tags_n = 10
    base = 90
    tags_inc = int((255 - base) / tags_n)
    for t in range(tags_n+1):
        tag = "tag" + str(t)
        tk_rgb = "#%02x%02x%02x" % (base + (tags_inc * t), 100, 255 - (tags_inc * t))
        root.txt.tag_config(tag, background=tk_rgb, foreground="black")
        root.txt.insert(tk.END, tag, tag)
    root.txt.tag_config("white", background="white", foreground="black")
    root.txt.insert(tk.END, "white\n\n", "white")
    print(cell_states.shape)
    print(len(txt))
    for i in range(cell_states.shape[1]):
        c = cell_states[:, i]
        c_n = (c - np.min(c)) / (np.max(c)-np.min(c))
        c_n = np.tanh(c) / 2 + 0.5
        c_n = abs(np.tanh(c))
        o = str(i)
        # o = ix_to_char[i]
        max_char = np.argmax(Why[:, i])
        max_char = idx_to_char[max_char]
        root.txt.insert(tk.END, "\nWeight " + o + ": (" + str(max_char) + ")\n", "white")
        for l in range(len(txt)):
            if txt[l] == '\n':
                tag = "white"
            else:
                tag_ix = int(tags_n * c_n[l])
                tag = "tag" + str(tag_ix)
                # print(tag)
            root.txt.insert(tk.END, txt[l], tag)
    root.mainloop()


def encode(seq):                            # 1-of-k encoding
    enc = np.zeros((1, input_dim), dtype=int)
    enc[0][seq[0]] = 1
    for i in range(1, len(seq)):
        row = np.zeros((1, input_dim), dtype=int)
        row[0][seq[i]] = 1
        enc = np.append(enc, row, axis=0)
    return enc


def sample(x, h, sample_len):
    text = []
    c = np.zeros_like(h)
    var_track = np.zeros_like(h)
    for i in range(sample_len):
        h, c, cache = lstm_step_forward(x, h, c, Wx, Wh, b)
        scores = np.dot(h, Why.T) + by
        p = np.exp(scores) / np.sum(np.exp(scores))
        out = np.random.choice(range(input_dim), p=p[0])
        # out = np.argmax(p)
        x = np.zeros(input_dim)
        x[out] = 1
        next_character = idx_to_char[out]
        text.append(next_character)
        _, _, _, _, _, _, _, i, f, o, g, _ = cache
        # print(c.shape)
        var_track = np.append(var_track, c, axis=0)
    txt = ''.join(char for char in text)
    return txt, var_track[1:]


data_name = 'shakespeare'
data_vars, hyperparams, model_vars = pickle.load(open(data_name + '_model.p', 'rb'), encoding='latin1')
hist_vars = pickle.load(open(data_name + '_hist.p', 'rb'), encoding='latin1')
input_file, char_to_idx, idx_to_char = data_vars
seq_length, input_dim, hidden_dim, learning_rate = hyperparams
Wx, Wh, Why, b, by = model_vars
it, loss, smooth_loss, el_time = hist_vars
# data, _, _, _ = load(input_file)


print(Wx.shape)
print(Wh.shape)
print(Why.shape)
print(b.shape)
print(by.shape)
prev_h = np.zeros((1, hidden_dim))
# prev_h = np.random.randn(1, hidden_dim)
print("---")
char = int(np.random.uniform(0, input_dim))
print(idx_to_char[char])
seed = encode([char])
text, neuron = sample(seed, prev_h, 500)
print(text)
print(neuron.shape)
print_tk_txt(text, neuron)
print("---")