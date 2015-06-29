import numpy as np
from theano.compat.python2x import OrderedDict
from kdl_template import *

# random state so script is deterministic
random_state = np.random.RandomState(1999)
# shared variable graph for this experiment
graph = OrderedDict()

# Dataset setup and cleaning
text = fetch_lovecraft()
(symbols, mapper_func,
 inverse_mapper_func, mapper) = make_character_level_from_text(text)
# limit lines to 100 characters
symbols = [s[:100] for s in symbols]

minibatch_size = 100

# Get next step prediction targets
X_clean = [s[:-1] for s in symbols]
y_clean = [s[1:] for s in symbols]
X_clean = even_slice(X_clean, minibatch_size)
y_clean = even_slice(y_clean, minibatch_size)

# Whole dataset is about 40,000 lines, use 80% training 20% valid split
X_clean_train = X_clean[:-8000]
y_clean_train = y_clean[:-8000]
X_clean_valid = X_clean[-8000:]
y_clean_valid = y_clean[-8000:]

text_vocab_size = len(mapper.keys())

# sample minibatch
indices = minibatch_indices(symbols, minibatch_size)
start, stop = indices[0]
text_minibatcher = gen_text_minibatch_func(text_vocab_size)
X_minibatch, X_mask = text_minibatcher(X_clean, start, stop)
y_minibatch, y_mask = text_minibatcher(y_clean, start, stop)

# Build model - every layer will be the same size except the output layer
n_hid = 512
n_out = text_vocab_size

# Example input datasets (where first dimension is time)
# Used for debug and shape setup purposes
datasets_list = [X_minibatch, X_mask, y_minibatch, y_mask]
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, ["X", "X_mask", "y", "y_mask"], graph,
    list_of_test_values=datasets_list)

# Dropout control switch (easiest to set with givens, but can take as input)
drop_on = as_shared(1., 'drop_on')

# Dropout_input
drop_inp = dropout_layer([X_sym], 'dropout_inp', drop_on, dropout_prob=0.2,
                         random_state=random_state)

proj_inp = linear_layer([drop_inp], graph, 'proj_inp', n_hid,
                        random_state=random_state)

# Recurrent layers
h1 = easy_gru_recurrent([proj_inp], X_mask_sym,
                        n_hid, graph, 'l1_rec', random_state, one_step=False)

# Dropout between recurrent layers
drop_h1 = dropout_layer([h1], 'dropout_h1',
                        drop_on,
                        dropout_prob=0.5,
                        random_state=random_state)

h2 = easy_gru_recurrent([drop_h1], X_mask_sym,
                        n_hid, graph, 'l2_rec', random_state, one_step=False)

drop_h2 = dropout_layer([h2], 'dropout_h2',
                        drop_on,
                        dropout_prob=0.5,
                        random_state=random_state)

h3 = easy_gru_recurrent([drop_h2], X_mask_sym,
                        n_hid, graph, 'l3_rec', random_state, one_step=False)
# Output layer
y_hat = softmax_layer([h3], graph, 'softmax_pred', n_out, random_state)

cost = categorical_crossentropy_nll(y_hat, y_sym)
cost = masked_cost(cost, y_mask_sym).mean()

# Parameters of the model
params, grads = get_params_and_grads(graph, cost)

# Use rmsprop with gradient clipping to optimize
opt = rmsprop(params)
learning_rate = 0.0001
updates = opt.updates(params, grads, learning_rate, momentum=0.99)


# Continue from checkpoint if it exists
save_path = "serialized_char_rnn.pkl"
if not os.path.exists(save_path):
    fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                   [cost], updates=updates,
                                   givens={drop_on: 1.},
                                   on_unused_input='warn')
    cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                    [cost],
                                    givens={drop_on: 1.},
                                    on_unused_input='warn')
    predict_function = theano.function([X_sym, X_mask_sym], [y_hat],
                                       givens={drop_on: 0.})
    checkpoint_dict = {}
    checkpoint_dict["fit_function"] = fit_function
    checkpoint_dict["cost_function"] = cost_function
    checkpoint_dict["predict_function"] = predict_function
    previous_epoch_results = None
else:
    checkpoint_dict = load_checkpoint(save_path)
    fit_function = checkpoint_dict["fit_function"]
    cost_function = checkpoint_dict["cost_function"]
    predict_function = checkpoint_dict["predict_function"]
    previous_epoch_results = checkpoint_dict["previous_epoch_results"]


# Early stopping status function
def status_func(status_number, epoch_number, epoch_results):
    valid_results = iterate_function(
        cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
        list_of_output_names=["valid_cost"],
        list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
        shuffle=False)
    early_stopping_status_func(valid_results["valid_cost"][-1],
                               save_path, checkpoint_dict, epoch_results)

# Main training loop
epoch_results = iterate_function(
    fit_function, [X_clean_train, y_clean_train], minibatch_size,
    list_of_output_names=["cost"],
    list_of_minibatch_functions=[text_minibatcher], n_epochs=100,
    previous_epoch_results=previous_epoch_results,
    status_func=status_func, shuffle=True, random_state=random_state)
