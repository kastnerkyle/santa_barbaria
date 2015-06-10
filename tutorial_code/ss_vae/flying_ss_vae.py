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
encode_function = checkpoint_dict["encode_function"]
decode_function = checkpoint_dict["decode_function"]
predict_function = checkpoint_dict["predict_function"]

random_state = np.random.RandomState(args.seed)
train, valid, test = fetch_binarized_mnist()
# visualize against validation so we aren't cheating
X = valid[0].astype(theano.config.floatX)
y = valid[1].astype("int32")

# number of samples
n_plot_samples = 6
n_classes = 10
# MNIST dimensions
width = 28
height = 28
# Get random data samples
ind = np.arange(len(X))
random_state.shuffle(ind)
sample_X = X[ind[:n_plot_samples]]
sample_y = convert_to_one_hot(y[ind[:n_plot_samples]], n_classes=n_classes)


def gen_samples(X, y):
    mu, log_sig = encode_function(X)
    # No noise at test time - repeat y twice because y_pred is needed for Theano
    # But it is not used unless y_sym is all -1
    out, = decode_function(mu + np.exp(log_sig), y, y.astype("float32"))
    return out

# VAE specific plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
generated_X = gen_samples(sample_X, sample_y)
y_hat, = predict_function(sample_X)

all_pred_y, = predict_function(X)
all_pred_y = np.argmax(all_pred_y, axis=1)
accuracy = np.mean(all_pred_y.ravel() == y.ravel())
pred_y = np.argmax(y_hat, axis=1).ravel()
true_y = np.argmax(sample_y, axis=1).ravel()
f, axarr = plt.subplots(n_plot_samples, 2)
for n, (X_i, y_i, sx_i, sy_i) in enumerate(zip(sample_X, true_y,
                                               generated_X, pred_y)):
    axarr[n, 0].matshow(X_i.reshape(width, height), cmap="gray")
    axarr[n, 1].matshow(sx_i.reshape(width, height), cmap="gray")
    axarr[n, 0].axis('off')
    axarr[n, 1].axis('off')
    axarr[n, 0].text(0, 7, str(y_i), color='green')
    if y_i == sy_i:
        axarr[n, 1].text(0, 7, str(sy_i), color='green')
    else:
        axarr[n, 1].text(0, 7, str(sy_i), color='red')

f.suptitle("Validation accuracy: %s" % str(accuracy))
plt.savefig('vae_reconstruction.png')
plt.close()

# Calculate noisy linear path between points in space
mus, log_sigmas = encode_function(sample_X)
n_steps = 20
mu_path = interpolate_between_points(mus, n_steps=n_steps)
log_sigma_path = interpolate_between_points(log_sigmas, n_steps=n_steps)

# Noisy path across space from one point to another
path_X = mu_path + np.exp(log_sigma_path)
path_y = np.zeros((len(path_X), n_classes), dtype="int32")

for i in range(n_plot_samples):
    path_y[i * n_steps:(i + 1) * n_steps] = sample_y[i]

# Have to pass another argument for y_pred
# But it is not used unless y_sym is all -1
out, = decode_function(path_X, path_y, path_y.astype("float32"))
text_y = [str(np.argmax(path_y[i])) for i in range(len(path_y))]
color_y = ["white"] * len(text_y)
make_gif(out, "vae_code.gif", width, height, list_text_per_frame=text_y,
         list_text_per_frame_color=color_y, delay=1, grayscale=True)
