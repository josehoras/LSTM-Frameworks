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

# Function to scan all operations in graph and filter by substring
def print_ops(string=""):
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    print("Total ops:", len(ops))
    for op in ops:
        if string in op.name:
            print(op.name, op.type)


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

# Bookkeeping variables
save_path = 'tensor_model/'

# Restore
print("Restoring the trained variables\n")
saver = tf.train.import_meta_graph(save_path + "model.meta")
print_ops(string="probabilities")

with tf.Session() as sess:
    # Restore graph tensors
    x = sess.graph.get_tensor_by_name('x:0')
    init_state = sess.graph.get_tensor_by_name('init_state:0')
    probabilities = sess.graph.get_tensor_by_name('probabilities:0')
    current_state_c = sess.graph.get_tensor_by_name('rnn/while/Exit_3:0')
    current_state_h = sess.graph.get_tensor_by_name('rnn/while/Exit_4:0')
    current_state = tf.nn.rnn_cell.LSTMStateTuple(current_state_c, current_state_h)
    # Initialize variables with saved values
    saver.restore(sess, save_path + "model")
    print(sample(600, sess))
