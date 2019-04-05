# Import all needed libraries
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time

# Load data
input_file = 'shakespeare.txt'
data = open(input_file, 'r').read() # should be simple plain text file
chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
print(chars)

# 1-of-k encoding
def encode(seq):
    enc = np.zeros((1, vocab_size), dtype=int)
    enc[0][seq[0]] = 1
    for i in range(1, len(seq)):
        row = np.zeros((1, vocab_size), dtype=int)
        row[0][seq[i]] = 1
        enc = np.append(enc, row, axis=0)
    return enc


# hyperparameters
seq_length = 5
hidden_dim = 64
batch_size = 1

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([hidden_dim, vocab_size]))
out_bias = tf.Variable(tf.random_normal([vocab_size]))

# model architecture
x = tf.placeholder("float", [seq_length, batch_size, vocab_size])
y = tf.placeholder("float", [seq_length, batch_size, vocab_size])
lstm_layer = rnn.LSTMCell(hidden_dim, forget_bias=1)
outputs, _ = tf.nn.dynamic_rnn(lstm_layer, x, dtype="float32")
outputs = tf.unstack(outputs, axis=0)
logits = [tf.matmul(output, out_weights) + out_bias for output in outputs]
probabilities = [tf.nn.softmax(logit) for logit in logits]

# model evaluation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
training_operation = optimizer.minimize(loss)

# same with chars to generate predictions
char_x = tf.placeholder("float", [1, batch_size, vocab_size])
char_out, _ = tf.nn.dynamic_rnn(lstm_layer, char_x, dtype="float32")
char_logits = tf.matmul(char_out[0], out_weights) + out_bias
char_prob = tf.nn.softmax(char_logits)

# training
p = 0
smooth_loss = (-np.log(1.0 / vocab_size) * seq_length) / seq_length  # loss at iteration 0
init = tf.global_variables_initializer()
with tf.Session() as sess:
    with tf.device("/device:CPU:0"):  # "/cpu:0" or "/gpu:0"
        start = time.time()
        print("* Second Session *")
        sess.run(init)
        for p in range(10001):
            if p + seq_length + 1 >= len(data) or p == 0:
                sess.run(lstm_layer.zero_state(batch_size, dtype=tf.float32)) # reset RNN memory
                p = 0 # go from start of data
            a = [char_to_ix[char] for char in data[p:p+seq_length]]  # Sequence of inputs (numbers)
            t = [char_to_ix[char] for char in data[p+1:p+1+seq_length]]
            inputs = np.expand_dims(encode(a), axis=1)
            targets = np.expand_dims(encode(t), axis=1)
            los, _ = sess.run([loss, training_operation], feed_dict={x: inputs, y: targets})
            smooth_loss = smooth_loss * 0.999 + los * 0.001
            if p % 5000 == 0:
                print()
                print(p, ": ", smooth_loss)
                seed = np.expand_dims(inputs[-1], axis=0)
                txt = ''
                for i in range(200):
                    probs = sess.run(char_prob, feed_dict={char_x: seed})
                    pred = np.random.choice(range(vocab_size), p=probs[0])
                    seed = np.expand_dims(encode([pred]), axis=0)
                    character = ix_to_char[pred]
                    txt = txt + character
                print(txt)
end = time.time()
print("      This thing has taken all this time: ", end - start)

