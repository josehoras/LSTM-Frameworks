# Import all needed libraries
import tensorflow as tf
import numpy as np
import time
import aux_funcs as aux

############
### Main ###
############
# load data
data, char_to_idx, idx_to_char, vocab_size = aux.load('shakespeare.txt')
print('data has %d characters, %d unique.' % (len(data), vocab_size))

# hyperparameters
learning_rate = 1e-2
seq_length = 100
hidden_dim = 500
batch_size = 1

# model architecture
def model(input_data, init_state):
    # lstm layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True)
    rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
    # affine layer
    out_weights = tf.get_variable("out_w", shape=[hidden_dim, vocab_size])
    out_bias = tf.get_variable("out_b", shape=[vocab_size])
    # model
    outputs, current_state = tf.nn.dynamic_rnn(lstm_cell, input_data, initial_state=rnn_tuple_state,
                                        time_major=True, dtype=tf.float32)
    logits = tf.matmul(outputs[:, 0, :], out_weights) + out_bias
    probabilities = tf.nn.softmax(logits, name="probabilities")
    return logits, probabilities, current_state


def opt(logits, y):
    # model evaluation
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training = optimizer.minimize(loss, name="training")
    return loss, training


def sample(sample_length, session):
    seed = aux.encode([int(np.random.uniform(0, vocab_size))], vocab_size)
    seed = np.array([seed])
    _char_state = np.zeros((2, batch_size, hidden_dim))
    txt = ''
    for i in range(sample_length):
        char_probs, _char_state = session.run([probs, current_state],
                                           feed_dict={x: seed, init_state: _char_state})
        pred = np.random.choice(range(vocab_size), p=char_probs[0])
        seed = np.expand_dims(aux.encode([pred], vocab_size), axis=0)
        character = idx_to_char[pred]
        txt = txt + character
    return txt


# TensorFlow input variables
x = tf.placeholder("float", [None, batch_size, vocab_size])
y = tf.placeholder("float", [None, batch_size, vocab_size])
init_state = tf.placeholder(tf.float32, [2, batch_size, hidden_dim])
# model outputs
logits, probs, current_state = model(x, init_state)
# optimization outputs
loss, training = opt(logits, y)

# history variables
loss_hist = [-np.log(1.0 / vocab_size)]  # loss at iteration 0
smooth_loss = loss_hist.copy()
it = 0
it_per_epoch = len(data) / seq_length
p = (it % it_per_epoch) * seq_length
data_feed = aux.gen(data, seq_length, char_to_idx, vocab_size, p=p)
elapsed_time = 0
# training
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _current_state = np.zeros((2, batch_size, hidden_dim))
    for p in range(4001):
        # show progress
        if p % 100 == 0:
            print('\niter %d, loss: %f' % (p, smooth_loss[-1]))  # print progress
            print(sample(600, sess))
            aux.plot(loss_hist, smooth_loss, it, it_per_epoch, base_name="tensor")
        # collect data for next step
        inputs, targets = (next(data_feed))
        l, _, _current_state = sess.run([loss, training, current_state],
                                        feed_dict={x: inputs,
                                                   y: targets,
                                                   init_state: _current_state})
        loss_hist.append(l)
        smooth_loss.append(smooth_loss[-1] * 0.999 + loss_hist[-1] * 0.001)

end = time.time()
print("      This thing has taken all this time: ", end - start, "\n")

