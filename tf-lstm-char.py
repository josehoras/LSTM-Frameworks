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


# prepare input
seq_length = 5
xn = [char_to_ix[char] for char in data[0:seq_length]]
tn = [char_to_ix[char] for char in data[1:1+seq_length]]
target = encode(tn)
target = np.expand_dims(target, axis=1)
print("xn shape: ", len(xn))
xn_enc = encode(xn)
print("inp shape: ", xn_enc.shape)
inp = np.expand_dims(xn_enc, axis=1)
print("inp shape: ", inp.shape)

hidden_dim = 64
batch_size = 1

x = tf.placeholder("float", [seq_length, batch_size, vocab_size])
y = tf.placeholder("float", [seq_length, batch_size, vocab_size])

print("x shape: ", x.shape)
input = tf.unstack(x, axis=0)

print("Input length: ", len(input), input[0].shape)

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([hidden_dim, vocab_size]))
out_bias = tf.Variable(tf.random_normal([vocab_size]))

lstm_layer = rnn.LSTMCell(hidden_dim, forget_bias=1)
# state = lstm_layer.zero_state(batch_size, dtype=tf.float32)
# i_state = lstm_layer.get_initial_state(batch_size=batch_size, dtype=tf.int32)
# print("state shape: ", len(state), state[0].shape, state[1].shape)
# print("state 0: ", state[0])
outputs, state = rnn.static_rnn(lstm_layer, input, dtype="float32")
print("output shape: ", len(outputs), outputs[0].shape)
logits = [tf.matmul(output, out_weights) + out_bias for output in outputs]
probabilities = [tf.nn.softmax(logit) for logit in logits]

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
training_operation = optimizer.minimize(loss)

# same with chars to generate predictions
char_x = tf.placeholder("float", [1, batch_size, vocab_size])
char_input = tf.unstack(char_x, axis=0)
char_out, _ = rnn.static_rnn(lstm_layer, char_input, dtype="float32")
char_logits = [tf.matmul(char_o, out_weights) + out_bias for char_o in char_out]
char_prob = [tf.nn.softmax(char_logit) for char_logit in char_logits]

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    print("* In Session *")
    sess.run(init)
    inpu = sess.run(input, feed_dict={x: inp})
    print("inpu shape: ", len(inpu), len(inpu[0]), len(inpu[0][0]))
    print("inpu 0: ", inpu[0][0])
    out = sess.run(outputs, feed_dict={x: inp})
    print("out shape: ", len(out), len(out[0]), out[0][0])
    f_state = sess.run(state, feed_dict={x: inp})
    print("f_state shape: ", len(f_state), len(f_state[0]))
    logs = sess.run(logits, feed_dict={x: inp})
    print("logs shape: ", len(logs), logs[0].shape)
    # print("logs: ", logs)
    probs = sess.run(probabilities, feed_dict={x: inp})
    # print("probs: ", probs)
    print("total probs: ", np.sum(probs))
    for prob in probs:
        pred = np.random.choice(range(vocab_size), p=prob[0])
        character = ix_to_char[pred]
        print(character, end='')
    print('\n')
    for i in range(seq_length):
        character = ix_to_char[xn[i]]
        print(character, end='')
    print('\n')
    for i in range(seq_length):
        character = ix_to_char[tn[i]]
        print(character, end='')
    print('\n')

# well I repeat her to do the training
p = 0
smooth_loss = (-np.log(1.0 / vocab_size) * seq_length) / seq_length  # loss at iteration 0
init=tf.global_variables_initializer()
with tf.Session() as sess:
    print("* Second Session *")
    sess.run(init)
    los = sess.run(loss, feed_dict={x: inp, y: target})
    print(los)
    for p in range(200000):
        if p + seq_length + 1 >= len(data) or p == 0:
            prev_h = np.zeros((1, hidden_dim)) # reset RNN memory
            p = 0 # go from start of data
        a = [char_to_ix[char] for char in data[p:p+seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_ix[char] for char in data[p+1:p+1+seq_length]]
        inputs = encode(a)
        inputs = np.expand_dims(inputs, axis=1)
        targets = encode(t)
        targets = np.expand_dims(targets, axis=1)
        los, _ = sess.run([loss, training_operation], feed_dict={x: inputs, y: targets})
        smooth_loss = smooth_loss * 0.999 + los * 0.001
        if p % 5000 == 0:
            print(p, ": ", smooth_loss)
            seed = np.expand_dims(inputs[-1], axis=0)
            txt = ''
            for i in range(200):
                probs = sess.run(char_prob, feed_dict={char_x: seed})
                pred = np.random.choice(range(vocab_size), p=probs[0][0])
                seed = np.expand_dims(encode([pred]), axis=0)
                character = ix_to_char[pred]
                print(character, end='')
            print('\n')


