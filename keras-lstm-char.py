# Import all needed libraries
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, TimeDistributed
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import numpy as np
import time
from keras import backend as K
from keras.models import load_model

sess = tf.Session()
K.set_session(sess)

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
seq_length = 100
hidden_dim = 64
batch_size = 1

# Training pipeline
x = tf.placeholder("float", [None, batch_size, vocab_size])
y = tf.placeholder("float", [None, batch_size, vocab_size])
layer = LSTM(hidden_dim, return_sequences=True)(x)
layer = LSTM(hidden_dim, return_sequences=True)(layer)
layer = TimeDistributed(Dense(vocab_size))(layer)
probabilities = Activation('softmax')(layer)
loss = tf.reduce_mean(categorical_crossentropy(y, probabilities))
acc_value = accuracy(y, probabilities)
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

saver = tf.train.Saver()
model_file = "./shakespeare_keras_model"

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
do_training = False
if do_training:
    with sess.as_default():
        p = 0
        it = 0
        for p in range(4001):
            if p + seq_length + 1 >= len(data):
                print("w")
                p = 0  # go to start of data
            a = [char_to_ix[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
            t = [char_to_ix[char] for char in data[p + 1: p + 1 + seq_length]]
            inputs = np.expand_dims(encode(a), axis=1)
            targets = np.expand_dims(encode(t), axis=1)
            train_step.run(feed_dict={x: inputs, y: targets})
            if it % 1000 == 0:
                print("Iteration: ", it, "Acc: ", np.sum(acc_value.eval(feed_dict={x: inputs, y: targets})))
                print("Loss: ", loss.eval(feed_dict={x: inputs, y: targets}))
                print("-----------------------------------")
                input_one = np.array([[inputs[0, -1]]])
                txt = ''
                for i in range(350):
                    prob = probabilities.eval(feed_dict={x: input_one})
                    pred = np.random.choice(range(vocab_size), p=prob[0][0])
                    character = ix_to_char[pred]
                    input_one = np.expand_dims(encode([pred]), axis=0)
                    txt = txt + character
                print(txt)
                print("-----------------------------------")
                print()

            p += 1
            it += 1
        all_vars = tf.trainable_variables()
        for i in range(len(all_vars)):
            name = all_vars[i].name
            values = sess.run(name)
            print('name', name)
            print('shape', values.shape)
        saver.save(sess, model_file)
        print("Model saved")


# Test
with tf.Session() as sess:
    print("-----------------------------------")
    print("* Test *")
    print("-----------------------------------")
    saver = tf.train.import_meta_graph('shakespeare_keras_model.meta')
    saver.restore(sess, model_file)
    pred = [int(np.random.uniform(0, vocab_size))]
    input_one = np.expand_dims(encode([pred]), axis=0)
    txt = ''
    for i in range(350):
        prob = probabilities.eval(feed_dict={x: input_one})
        pred = np.random.choice(range(vocab_size), p=prob[0][0])
        character = ix_to_char[pred]
        input_one = np.expand_dims(encode([pred]), axis=0)
        txt = txt + character
    print(txt)
    print("-----------------------------------")
    print()


# Full Keras
# model = Sequential()
# model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(1, look_back)))
# model.add(TimeDistributed(Dense(vocab_size)))
# model.add(Activation('softmax'))
