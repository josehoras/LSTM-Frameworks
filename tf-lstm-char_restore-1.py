# Import all needed libraries
import tensorflow as tf
import numpy as np
import time
import aux_funcs as aux


# Function to sample text using a model trained on a certain text corpus
def sample(sample_length, session):
    seed = aux.encode([int(np.random.uniform(0, vocab_size))], vocab_size)
    seed = np.array([seed])
    _char_state = np.zeros((2, batch_size, hidden_dim))
    txt = ''
    for i in range(sample_length):
        char_probs, _char_state = session.run([probabilities, current_state],
                                           feed_dict={x: seed, init_state: _char_state})
        pred = np.random.choice(range(vocab_size), p=char_probs[0])
        seed = np.expand_dims(aux.encode([pred], vocab_size), axis=0)
        character = idx_to_char[pred]
        txt += character
    return txt


############
### Main ###
############
# hyperparameters
learning_rate = 1e-2
seq_length = 100
hidden_dim = 500
batch_size = 1

# load data
data_name = 'shakespeare'
input_file = data_name +'.txt'
_, char_to_idx, idx_to_char, vocab_size = aux.load(input_file)
print("First 4 characters are: ", idx_to_char[0], idx_to_char[1], idx_to_char[2], idx_to_char[3])

# TensorFlow input variables
x = tf.placeholder("float", [None, batch_size, vocab_size], name="x")
y = tf.placeholder("float", [None, batch_size, vocab_size], name="y")
init_state = tf.placeholder(tf.float32, [2, batch_size, hidden_dim], name="init_state")
# model architecture
# lstm layer
lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True, name="celula")
rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
# dense layer parameters
dense_weights = tf.get_variable("out_w", shape=[hidden_dim, vocab_size])
dense_bias = tf.get_variable("out_b", shape=[vocab_size])
# model
h_states, current_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=rnn_tuple_state,
                                            time_major=True, dtype=tf.float32)
logits = tf.matmul(h_states[:, 0, :], dense_weights) + dense_bias
probabilities = tf.nn.softmax(logits, name="probabilities")

# model evaluation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training = optimizer.minimize(loss, name="training")

# Bookkeeping variables
save_path = 'tensor_model/'

# Saver
saver = tf.train.Saver()

# Restore
print("Restoring the trained variables\n")

with tf.Session() as sess:
    # If saver.restore(...) commented -> 'Error: Attempting to use uninitialized value out_w'
    # can be corrected with sess.run(init_op), but training is forgotten
    saver.restore(sess, save_path + "model")
    print(sample(600, sess))
