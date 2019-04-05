# Import all needed libraries
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Load data
input_file = 'shakespeare.txt'
data = open(input_file, 'r').read() # should be simple plain text file
chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))


def encode(seq):
    enc = np.zeros((1, vocab_size), dtype=int)
    enc[0][seq[0]] = 1
    for i in range(1, len(seq)):
        row = np.zeros((1, vocab_size), dtype=int)
        row[0][seq[i]] = 1
        enc = np.append(enc, row, axis=0)
    return enc

seq_length = 3
xn = [char_to_ix[char] for char in data[0:seq_length]]
print("xn shape: ", len(xn))
inp = encode(xn)
print("inp shape: ", inp.shape)
inp = np.expand_dims(inp, axis=0)
print("inp shape: ", inp.shape)

num_units = 10

x=tf.placeholder("float", [None, seq_length, vocab_size])
print("x shape: ", x.shape)
input = tf.unstack(x, num=seq_length, axis=1)

lstm_layer = rnn.LSTMCell(num_units,forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

print(outputs)

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    inpu = sess.run(input, feed_dict={x: inp})
    print("inpu shape: ", len(inpu), len(inpu[0]), len(inpu[0][0]))
    print(inpu[0][0])
    out = sess.run(outputs, feed_dict={x: inp})
    print("out shape: ", len(out), len(out[0]), out[0][0])





# Define model architecture
# seq_length = 100
# p = 0
# while True:
#     # Go through the RNN
#     if p + seq_length + 1 >= len(data) or p == 0:
#         p = 0 # go from start of data
#     x = [char_to_ix[char] for char in data[p:p+seq_length]]  # Sequence of inputs (numbers)
#     t = [char_to_ix[char] for char in data[p+1:p+1+seq_length]]
#     inputs = encode(x)
#     targets = encode(t)
