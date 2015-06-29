from kdl_template import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("saved_functions_file",
                    help="Saved pickle file from vae training")
parser.add_argument("--seed", "-s",
                    help="random seed for path calculation",
                    action="store", default=1979, type=int)

args = parser.parse_args()
if not os.path.exists(args.saved_functions_file):
    raise ValueError("Please provide a valid path for saved pickle file!")

checkpoint_dict = load_checkpoint(args.saved_functions_file)
predict_function = checkpoint_dict["predict_function"]
epoch_results = checkpoint_dict["previous_epoch_results"]

random_state = np.random.RandomState(args.seed)

text = fetch_lovecraft()
(symbols, mapper_func,
 inverse_mapper_func, mapper) = make_character_level_from_text(text)
symbols = [s[:100] for s in symbols]

minibatch_size = 100

# Get next step prediction targets
X_clean = [s[:-1] for s in symbols]
y_clean = [s[1:] for s in symbols]
X_clean = even_slice(X_clean, minibatch_size)
y_clean = even_slice(y_clean, minibatch_size)
X_clean_train = X_clean[:-1000]
y_clean_train = y_clean[:-1000]
X_clean_valid = X_clean[-1000:]
y_clean_valid = y_clean[-1000:]
text_vocab_size = len(mapper.keys())
indices = minibatch_indices(symbols, minibatch_size)

start, stop = indices[0]
text_minibatcher = gen_text_minibatch_func(text_vocab_size)
X_minibatch, X_mask = text_minibatcher(X_clean, start, stop)
y_minibatch, y_mask = text_minibatcher(y_clean, start, stop)
