# Import all needed libraries
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import aux_funcs as aux
import os
import shutil
from tensorflow.python.saved_model import tag_constants

def gen(data, seq_length, p=0):
    p = int(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            prev_h = np.zeros((1, hidden_dim))  # reset LSTM memory
            p = 0  # go to start of data
        a = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = np.expand_dims(aux.encode(a, vocab_size), axis=1)  # shape: (seq_length, input_dim)
        targets = np.expand_dims(aux.encode(t, vocab_size), axis=1)
        p = p + seq_length
        yield inputs, targets


############
### Main ###
############
if tf.test.is_gpu_available():
    print("GPU!")
else:
    print("NO GPU!")

# tf.reset_default_graph()

# load data
data, char_to_idx, idx_to_char, vocab_size = aux.load('shakespeare.txt')
print('data has %d characters, %d unique.' % (len(data), vocab_size))

# hyperparameters
learning_rate = 1e-2
seq_length = 100
hidden_dim = 500
batch_size = 1


# model architecture
def model(inputs_data):
    # lstm layer
    lstm_layer = tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True, name="lstm")
    # affine layer
    out_weights = tf.get_variable("out_w", shape=[hidden_dim, vocab_size])
    out_bias = tf.get_variable("out_b", shape=[vocab_size])
    rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
    outputs, current_state = tf.nn.dynamic_rnn(lstm_layer, inputs_data, initial_state=rnn_tuple_state,
                                        time_major=True, dtype=tf.float32)
    outputs = outputs[:, 0, :]  # here we use always batch_size=1
    logits = tf.matmul(outputs, out_weights) + out_bias
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


def sample(sample_length, session, probabilites, state):
    seed = aux.encode([int(np.random.uniform(0, vocab_size))], vocab_size)
    seed = np.array([seed])
    _char_state = np.zeros((2, batch_size, hidden_dim))
    txt = ''
    for i in range(sample_length):
        char_probs, _char_state = session.run([probabilites, state],
                                           feed_dict={x: seed, init_state: _char_state})
        pred = np.random.choice(range(vocab_size), p=char_probs[0])
        seed = np.expand_dims(aux.encode([pred], vocab_size), axis=0)
        character = idx_to_char[pred]
        txt = txt + character
    return txt


# history variables
loss_hist = [-np.log(1.0 / vocab_size)]  # loss at iteration 0
smooth_loss = loss_hist.copy()
it = 0
it_per_epoch = len(data) / seq_length
data_feed = gen(data, seq_length, p=(it % it_per_epoch) * seq_length)
elapsed_time = 0

# training
model_dir = 'shakespeare'
start = time.time()
graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:
        print("* Session *")
        load_model = True
        if load_model:
            # sess.run(tf.global_variables_initializer())
            print("Loading model")
            cwd = os.getcwd()
            path = os.path.join(cwd, model_dir)
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], path)
            values = sess.run('rnn/lstm/kernel:0')
            print("Que valor: ", values[3, 1000])
            # graph = tf.get_default_graph()
            # input variables
            x = graph.get_tensor_by_name('Placeholder:0')
            y = graph.get_tensor_by_name('Placeholder_1:0')
            init_state = graph.get_tensor_by_name('Placeholder_2:0')
            # model
            out_weights = graph.get_tensor_by_name('out_w:0')
            out_bias = graph.get_tensor_by_name('out_b:0')
            probs = graph.get_tensor_by_name('probabilities:0')
            current_state = tf.stack([graph.get_tensor_by_name('rnn/while/Exit_3:0'),
                                      graph.get_tensor_by_name('rnn/while/Exit_4:0')], 0)
            # optimization
            loss = graph.get_tensor_by_name('loss:0')
            training = graph.get_operation_by_name('training')
        else:
            print("Starting new")
            # input variables
            x = tf.placeholder("float", [None, batch_size, vocab_size])
            y = tf.placeholder("float", [None, batch_size, vocab_size])
            init_state = tf.placeholder(tf.float32, [2, batch_size, hidden_dim])
            # model
            logits, probs, current_state = model(x)
            # optimization
            loss, training = opt(logits, y)
            sess.run(tf.global_variables_initializer())

        _current_state = np.zeros((2, batch_size, hidden_dim))
        for p in range(101):
            # show progress
            if p % 100 == 0:
                print('\niter %d, loss: %f' % (p, smooth_loss[-1]))  # print progress
                print(sample(600, sess, probs, current_state))
                aux.plot(loss_hist, smooth_loss, it, it_per_epoch, base_name="tensor")
            # collect data for next step
            inputs, targets = (next(data_feed))
            l, _, _current_state = sess.run([loss, training, current_state],
                                            feed_dict={x: inputs,
                                                       y: targets,
                                                       init_state: _current_state})
            loss_hist.append(l)
            smooth_loss.append(smooth_loss[-1] * 0.999 + loss_hist[-1] * 0.001)
        values = sess.run('rnn/lstm/kernel:0')
        print("Que valor: ", values[3, 1000])
        # Saving
        print('\nSaving...')
        cwd = os.getcwd()
        path = os.path.join(cwd, model_dir)
        shutil.rmtree(path, ignore_errors=True)
        inputs_dict = {"x": x,
                       "y": y,
                       "init_state": init_state}
        outputs_dict = {
            "probs": probs,
            "loss": loss
        }
        tf.saved_model.simple_save(sess, path, inputs_dict, outputs_dict)
        print('Ok')
end = time.time()
print("      This thing has taken all this time: ", end - start, "\n")

# Restoring
graph2 = tf.Graph()
with graph2.as_default():
    with tf.Session(graph= graph2) as sess2:
        # Restore saved values
        print('\nRestoring...')
        tf.saved_model.loader.load(sess2, [tag_constants.SERVING], path)
        print('Ok')
        values = sess2.run('rnn/lstm/kernel:0')
        print("Que valor: ", values[3, 1000])
        all_vars = tf.trainable_variables()
        print(len(all_vars))
        for i in range(len(all_vars)):
            name = all_vars[i].name
            values = sess2.run(name)
            print('name', name)
            print('shape', values.shape)
        # keys = graph.get_all_collection_keys()
        # print(keys)
        # for op in graph.get_operations():
        #     print(op.name, graph.get_tensor_by_name(op.name + ':0').shape)
        # scope = graph.get_name_scope()
        # print(scope)
        # col = graph.get_collection(scope)
        # print(col)
        # graph = tf.get_default_graph()

        x = graph2.get_tensor_by_name('Placeholder:0')
        y = graph2.get_tensor_by_name('Placeholder_1:0')
        init_state = graph2.get_tensor_by_name('Placeholder_2:0')
        probs_ = graph2.get_tensor_by_name('probabilities:0')
        current_state_c = graph2.get_tensor_by_name('rnn/while/Exit_3:0')
        current_state_h = graph2.get_tensor_by_name('rnn/while/Exit_4:0')
        current_state_ = tf.stack([current_state_c, current_state_h], 0)
        loss_ = graph2.get_tensor_by_name('loss:0')
        training_ = graph2.get_operation_by_name('training')

        for p in range(101):
            # show progress
            if p % 100 == 0:
                print('\niter %d, loss: %f' % (p, smooth_loss[-1]))  # print progress
                print(sample(600, sess2, probs_, current_state_))
                aux.plot(loss_hist, smooth_loss, it, it_per_epoch, base_name="tensor")
            # collect data for next step
            inputs, targets = (next(data_feed))
            l, _, _current_state = sess2.run([loss_, training_, current_state_],
                                            feed_dict={x: inputs,
                                                       y: targets,
                                                       init_state: _current_state})
            loss_hist.append(l)
            smooth_loss.append(smooth_loss[-1] * 0.999 + loss_hist[-1] * 0.001)


        print("loss: ", l)
        print(sample(600, sess2, probs_, current_state_))
        values = sess2.run('rnn/lstm/kernel:0')
        print("Que valor: ", values[3, 1000])
        # Saving
        print('\nSaving...')
        cwd = os.getcwd()
        path = os.path.join(cwd, model_dir)
        shutil.rmtree(path, ignore_errors=True)
        inputs_dict = {"x": x,
                       "y": y,
                       "init_state": init_state}
        outputs_dict = {
            "probs": probs,
            "loss": loss
        }
        tf.saved_model.simple_save(sess2, path, inputs_dict, outputs_dict)
        print('Ok')