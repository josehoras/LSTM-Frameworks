# Import all needed libraries
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
from time import time
import aux_funcs as aux
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, TimeDistributed
from keras.models import load_model
from keras.callbacks import LambdaCallback, Callback
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

from keras.models import Model
from keras.layers import Input


# Keras callbacks
def test(epoch, logs):
    print("-----------------------------------")
    print("* Test *")
    print(epoch, logs)
    print("-----------------------------------")
    txt = ''
    seed = aux.encode([int(np.random.uniform(0, vocab_size))], vocab_size)
    seed = np.array([seed])
    init_state_h = np.zeros((1, batch_size, hidden_dim))
    init_state_c = np.zeros((1, batch_size, hidden_dim))
    # print(seed.shape)
    for i in range(400):
        prob, init_state_h, init_state_c = pred_model.predict([seed, init_state_h, init_state_c])
        # print(len(prob))
        pred = np.random.choice(range(vocab_size), p=prob[-1][0])
        # print("pred: ", pred)
        character = idx_to_char[pred]
        # print("character: ", character)
        seed = np.expand_dims(aux.encode([pred], vocab_size), axis=0)
        txt = txt + character
    print(txt)



class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = [-np.log(1.0 / vocab_size)]
        self.smooth_loss = [-np.log(1.0 / vocab_size)]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.smooth_loss.append(self.smooth_loss[-1] * 0.999 + logs.get('loss') * 0.001)
        if batch % 1000 == 0:
            print(batch, " ", self.smooth_loss[-1])
            test(0, logs)
            aux.plot(self.losses, self.smooth_loss, it, it_per_epoch, base_name="keras")


# load data
data, char_to_idx, idx_to_char, vocab_size = aux.load('shakespeare.txt')
print('data has %d characters, %d unique.' % (len(data), vocab_size))

# hyperparameters
learning_rate = 1e-2
seq_length = 100
hidden_dim = 500
batch_size = 1
epochs = 5

# instantiate generator
it = 0
reduce = 1/seq_length
it_per_epoch = np.int(len(data) / (seq_length*reduce))
p = (it % it_per_epoch) * seq_length
data_feed = aux.keras_gen(data, seq_length, char_to_idx, vocab_size, p=p)

# time counting starting here
t0 = time()

# Define Keras sequential model
# callbacks
test_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: test(epoch, logs))
history = LossHistory()

# Model API
inputs = Input(shape=(None, vocab_size))
lstm_layer = LSTM(hidden_dim, return_sequences=True, return_state=True)
lstm_output, _, _ = lstm_layer(inputs)
dense_layer = Dense(vocab_size, activation='softmax')
probabilities = dense_layer(lstm_output)
model = Model(inputs=inputs, outputs=probabilities)


state_input_h = Input(shape=(1, hidden_dim))
state_input_c = Input(shape=(1, hidden_dim))
states_inputs = [state_input_h, state_input_c]
outputs, state_h, state_c = lstm_layer(inputs, initial_state=[state_input_h, state_input_c])
states = [state_h, state_c]
pred_outputs = dense_layer(outputs)
pred_model = Model(inputs=[inputs, state_input_h, state_input_c], outputs=[pred_outputs, state_h, state_c])


# Sequential model
# model = Sequential()
# model.add(LSTM(hidden_dim, return_sequences=True, use_bias=True, return_state=False,
#                input_shape=(None, 67)))
# model.add(TimeDistributed(Dense(vocab_size)))
# model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
print(model.summary())
epochs_log = model.fit_generator(data_feed, steps_per_epoch=it_per_epoch, shuffle=False,
                                 epochs=epochs, callbacks=[test_callback, history], verbose=0)

# final time
print("Total time was: ", time() - t0)

# loss history plot
it = it_per_epoch * epochs
aux.plot(history.losses, history.smooth_loss, it, it_per_epoch, base_name="keras")

# Save model file
# model.save('model.h5')


