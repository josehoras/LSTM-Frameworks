import numpy as np


def sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))


# LSTM functions
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (1, input_dim)
    - prev_h: Previous hidden state, of shape (1, hidden_dim)
    - prev_c: previous cell state, of shape (1, hidden_dim)
    - Wx: Input-to-hidden weights, of shape (input_dim, 4*hidden_dim)
    - Wh: Hidden-to-hidden weights, of shape (hidden_dim, 4*hidden_dim)
    - b: Biases, of shape (4*hidden_dim)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    _, H = prev_h.shape
    a = prev_h.dot(Wh) + x.dot(Wx) + b      # (1, 4*hidden_dim)
    i = sigmoid(a[:, 0:H])
    f = sigmoid(a[:, H:2*H])
    o = sigmoid(a[:, 2*H:3*H])
    g = np.tanh(a[:, 3*H:4*H])              # (1, hidden_dim)
    next_c = f * prev_c + i * g             # (1, hidden_dim)
    next_h = o * (np.tanh(next_c))          # (1, hidden_dim)
    cache = x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c
    return next_h, next_c, cache


def lstm_forward(x, prev_h, Wx, Wh, b):
    """
    Inputs:
    - x: Input data of shape (seq_length, input_dim)
    - h0: Initial hidden state of shape (1, hidden_dim)
    - Wx: Weights for input-to-hidden connections, of shape (input_dim, 4*hidden_dim)
    - Wh: Weights for hidden-to-hidden connections, of shape (hidden_dim, 4*hidden_dim)
    - b: Biases of shape (4*hidden_dim)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (seq_length, hidden_dim)
    - cache: Values needed for the backward pass.
    """
    cache = []
    prev_c = np.zeros_like(prev_h)
    for i in range(x.shape[0]):     # 0 to seq_length-1
        next_h, next_c, next_cache = lstm_step_forward(x[i][None], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        cache.append(next_cache)
        if i > 0:
            h = np.append(h, next_h, axis=0)
        else:
            h = next_h
    return h, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c = cache
    _, H = dnext_h.shape
    d1 = o * (1 - np.tanh(next_c) ** 2) * dnext_h + dnext_c
    dprev_c = f * d1
    dop = np.tanh(next_c) * dnext_h
    dfp = prev_c * d1
    dip = g * d1
    dgp = i * d1
    do = sigmoid(a[:, 2*H:3*H]) * (1-sigmoid(a[:, 2*H:3*H])) * dop
    df = sigmoid(a[:, H:2*H]) * (1-sigmoid(a[:, H:2*H])) * dfp
    di = sigmoid(a[:, 0:H]) * (1-sigmoid(a[:, 0:H])) * dip
    dg = (1 - np.tanh(a[:, 3*H:4*H]) ** 2) * dgp
    da = np.concatenate((di, df, do, dg), axis=1)
    db = np.sum(da, axis=0)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H) (seq_length, hidden_dim)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    # print(dh.shape)
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, H = dh.shape
    dh_prev = 0
    dc_prev = 0
    for i in reversed(range(N)):
        dx_step, dh0_step, dc_step, dWx_step, dWh_step, db_step = lstm_step_backward(dh[i][None] + dh_prev, dc_prev, cache[i])
        dh_prev = dh0_step
        dc_prev = dc_step
        if i==N-1:
            dx = dx_step
            dWx = dWx_step
            dWh = dWh_step
            db = db_step
        else:
            dx = np.append(dx_step, dx, axis=0)
            dWx += dWx_step
            dWh += dWh_step
            db += db_step
    dh0 = dh0_step
    return dx, dh0, dWx, dWh, db

