import numpy as np
from theano.compat.python2x import OrderedDict
from kdl_template import *

# random state so script is deterministic
random_state = np.random.RandomState(1999)
# home of the computational graph
graph = OrderedDict()

# minibatch size
minibatch_size = 20
# number of input units
n_in = 5
# number of hidden units
n_hid = 10
# number of output units
n_out = 5

# Generate sinewaves offset in phase
n_timesteps = 50
d1 = 3 * np.arange(n_timesteps) / (2 * np.pi)
d2 = 3 * np.arange(n_in) / (2 * np.pi)
all_sines = np.sin(np.array([d1] * n_in).T + d2)

# Setup dataset and initial hidden vector of zeros
X = all_sines[:-1].astype(theano.config.floatX)
y = all_sines[1:].astype(theano.config.floatX)
h0 = np_zeros((n_hid,)).astype(theano.config.floatX)
c0 = np_zeros((n_hid,)).astype(theano.config.floatX)

# input (where first dimension is time)
X_sym, y_sym, h0_sym, c0_sym = add_datasets_to_graph([X, y, h0, c0],
                                                     ["X", "y", "h0", "c0"],
                                                     graph)

# Setup weights
proj_X = linear_layer([X_sym], graph, 'l1_proj', n_hid, random_state)


def step(x_t, h_tm1, c_tm1):
    h_t, c_t = lstm_recurrent_layer([x_t], [h_tm1], [c_tm1], graph, 'l1_rec',
                                    random_state)
    return h_t, c_t

# the hidden state `h` for the entire sequence
[h, c], _ = rnn_scan_wrap(step, name='main_scan', sequences=[proj_X],
                          outputs_info=[h0_sym, c0_sym])

# linear output activation
y_hat = linear_layer([h], graph, 'l2_proj', n_out, random_state)

# error between output and target
cost = ((y_sym - y_hat) ** 2).sum()
# Parameters of the model
params, grads = get_params_and_grads(graph, cost)

# Use stochastic gradient descent to optimize
opt = sgd(params)
learning_rate = 0.001
updates = opt.updates(params, grads, learning_rate)

# By returning h we can train while preserving hidden state from previous
# samples. This can allow for truncated backprop through time (TBPTT)!
fit_function = theano.function([X_sym, y_sym, h0_sym, c0_sym], [cost],
                               updates=updates)


def status_func(status_number, epoch_number, epoch_results):
    print_status_func(epoch_results)

epoch_results = iterate_function(fit_function, [X, y], minibatch_size,
                                 list_of_non_minibatch_args=[h0, c0],
                                 list_of_output_names=["cost"],
                                 n_epochs=2000,
                                 status_func=status_func,
                                 shuffle=True,
                                 random_state=random_state)
