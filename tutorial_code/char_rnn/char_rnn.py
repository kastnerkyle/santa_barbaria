import numpy as np
from theano.compat.python2x import OrderedDict
from kdl_template import *

# random state so script is deterministic
random_state = np.random.RandomState(1999)
# home of the computational graph
graph = OrderedDict()

text = fetch_lovecraft()
(symbols, mapper_func,
 inverse_mapper_func, mapper) = make_character_level_from_text(text)
symbols = [s[:100] for s in symbols]

minibatch_size = 500

# Get next step prediction targets
X_clean = [s[:-1] for s in symbols]
y_clean = [s[1:] for s in symbols]
X_clean = even_slice(X_clean, minibatch_size)
y_clean = even_slice(y_clean, minibatch_size)

text_vocab_size = len(mapper.keys())

# sample minibatch
indices = minibatch_indices(symbols, minibatch_size)
start, stop = indices[0]
text_minibatcher = gen_text_minibatch_func(text_vocab_size)
X_minibatch, X_mask = text_minibatcher(X_clean, start, stop)
y_minibatch, y_mask = text_minibatcher(y_clean, start, stop)

n_hid = 1000
n_out = text_vocab_size

# input (where first dimension is time)
datasets_list = [X_minibatch, X_mask, y_minibatch, y_mask]
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, ["X", "X_mask", "y", "y_mask"], graph,
    list_of_test_values=datasets_list)

# Setup weights
proj_X = linear_layer([X_sym], graph, 'l1_proj', n_hid, random_state)

# Recurrent
h = easy_gru_recurrent([proj_X], X_mask_sym,
                       n_hid, graph, 'l1_rec', random_state, one_step=False)

y_hat = softmax_layer([h], graph, 'l2_proj', n_out, random_state)
cost = categorical_crossentropy_nll(y_hat, y_sym)
cost = masked_cost(cost, y_mask_sym).mean()

# Parameters of the model
params, grads = get_params_and_grads(graph, cost)

# Use stochastic gradient descent to optimize
opt = rmsprop(params)
learning_rate = 0.001
updates = opt.updates(params, grads, learning_rate, momentum=0.9)

# By returning h we can train while preserving hidden state from previous
# samples. This can allow for truncated backprop through time (TBPTT)!
fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost],
                               updates=updates, on_unused_input='warn')


def status_func(status_number, epoch_number, epoch_results):
    print_status_func(epoch_results)

epoch_results = iterate_function(
    fit_function, [X_clean, y_clean], minibatch_size,
    list_of_output_names=["cost"],
    list_of_minibatch_functions=[text_minibatcher], n_epochs=100,
    status_func=status_func, shuffle=True, random_state=random_state)
