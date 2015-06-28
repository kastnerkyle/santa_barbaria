# Author: Kyle Kastner
# License: BSD 3-clause
# Ideas from Junyoung Chung and Kyunghyun Cho
# See https://github.com/jych/cle for a library in this style
import numpy as np
from scipy import linalg
from functools import reduce
import numbers
import random
import theano
import zipfile
import gzip
import os
import glob
import sys
import subprocess
try:
    import cPickle as pickle
except ImportError:
    import pickle
from theano import tensor
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import defaultdict


class sgd(object):
    def __init__(self, params):
        pass

    def updates(self, params, grads, learning_rate):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - learning_rate * grad))
        return updates


class sgd_nesterov(object):
    """
    Based on example from Yann D.
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
        not_finite = tensor.or_(tensor.isnan(grad_norm),
                                tensor.isinf(grad_norm))
        grad_norm = tensor.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = tensor.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = tensor.switch(not_finite, 0.1 * param,
                                 grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * tensor.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = tensor.sqrt(new_square - new_avg ** 2)
            rms_grad = tensor.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class adagrad(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, eps=1E-8):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            m_t = memory + grad ** 2
            g_t = grad / (eps + tensor.sqrt(m_t))
            p_t = param - learning_rate * g_t
            updates.append((memory, m_t))
            updates.append((param, p_t))
        return updates


class adam(object):
    """
    Based on implementation from @NewMu / Alex Radford
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        self.velocity_ = [theano.shared(np.zeros_like(p.get_value()))
                          for p in params]
        self.itr_ = theano.shared(np.array(0.).astype(theano.config.floatX))

    def updates(self, params, grads, learning_rate, b1=0.1, b2=0.001, eps=1E-8):
        updates = []
        itr = self.itr_
        i_t = itr + 1.
        fix1 = 1. - (1. - b1) ** i_t
        fix2 = 1. - (1. - b2) ** i_t
        lr_t = learning_rate * (tensor.sqrt(fix2) / fix1)
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            velocity = self.velocity_[n]
            m_t = (b1 * grad) + ((1. - b1) * memory)
            v_t = (b2 * tensor.sqr(grad)) + ((1. - b2) * velocity)
            g_t = m_t / (tensor.sqrt(v_t) + eps)
            p_t = param - (lr_t * g_t)
            updates.append((memory, m_t))
            updates.append((velocity, v_t))
            updates.append((param, p_t))
        updates.append((itr, i_t))
        return updates


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    if not data_dir:
        data_dir = os.getenv("SANTA_BARBARIA_DATA", os.path.join(
            os.path.expanduser("~"), "santa_barbaria_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def check_fetch_lovecraft():
    url = 'https://dl.dropboxusercontent.com/u/15378192/lovecraft_fiction.zip'
    partial_path = get_dataset_dir("lovecraft")
    full_path = os.path.join(partial_path, "lovecraft_fiction.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def make_character_level_from_text(text):
    all_chars = reduce(lambda x, y: set(x) | set(y), text, set())
    mapper = {k: n + 2 for n, k in enumerate(list(all_chars))}
    # 1 is EOS
    mapper["EOS"] = 1
    # 0 is UNK/MASK - unused here but needed in general
    mapper["UNK"] = 0
    inverse_mapper = {v: k for k, v in mapper.items()}

    def mapper_func(text_line):
        return [mapper[c] for c in text_line] + [mapper["EOS"]]

    def inverse_mapper_func(symbol_line):
        return "".join([inverse_mapper[s] for s in symbol_line
                        if s != mapper["EOS"]])

    # Remove blank lines
    cleaned = [mapper_func(t) for t in text if t != ""]
    return cleaned, mapper_func, inverse_mapper_func, mapper


def fetch_lovecraft():
    """ Returns lovecraft text. """
    data_path = check_fetch_lovecraft()
    all_data = []
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            all_data.extend(data)
    return all_data


def check_fetch_mnist():
    # py3k version is available at mnist_py3k.pkl.gz ... might need to fix
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    partial_path = get_dataset_dir("mnist")
    full_path = os.path.join(partial_path, "mnist.pkl.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_mnist():
    """ Returns mnist digits with picel values in [0 - 1] """
    data_path = check_fetch_mnist()
    f = gzip.open(data_path, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()
    return train_set, valid_set, test_set


def check_fetch_binarized_mnist():
    raise ValueError("Binarized MNIST has no labels!")
    url = "https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz"
    partial_path = get_dataset_dir("binarized_mnist")
    fname = "binarized_mnist.npz"
    full_path = os.path.join(partial_path, fname)
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    """
    # Personal version
    url = "https://dl.dropboxusercontent.com/u/15378192/binarized_mnist_%s.npy"
    fname = "binarized_mnist_%s.npy"
    for s in ["train", "valid", "test"]:
        full_path = os.path.join(partial_path, fname % s)
        if not os.path.exists(partial_path):
            os.makedirs(partial_path)
        if not os.path.exists(full_path):
            download(url % s, full_path, progress_update_percentage=1)
    """
    return partial_path


def fetch_binarized_mnist():
    train_set, valid_set, test_set = fetch_mnist()
    train_X = train_set[0]
    train_y = train_set[1]
    valid_X = valid_set[0]
    valid_y = valid_set[1]
    test_X = test_set[0]
    test_y = test_set[1]

    random_state = np.random.RandomState(1999)

    def get_sampled(arr):
        # make sure that a pixel can always be turned off
        return random_state.binomial(1, arr * 255 / 256., size=arr.shape)

    train_X = get_sampled(train_X)
    valid_X = get_sampled(valid_X)
    test_X = get_sampled(test_X)

    train_set = (train_X, train_y)
    valid_set = (valid_X, valid_y)
    test_set = (test_X, test_y)

    """
    # Old version for true binarized mnist
    data_path = check_fetch_binarized_mnist()
    fpath = os.path.join(data_path, "binarized_mnist.npz")

    arr = np.load(fpath)
    train_x = arr['train_data']
    valid_x = arr['valid_data']
    test_x = arr['test_data']
    train, valid, test = fetch_mnist()
    train_y = train[1]
    valid_y = valid[1]
    test_y = test[1]
    train_set = (train_x, train_y)
    valid_set = (valid_x, valid_y)
    test_set = (test_x, test_y)
    """
    return train_set, valid_set, test_set


def make_gif(arr, gif_name, plot_width, plot_height, list_text_per_frame=None,
             list_text_per_frame_color=None,
             delay=1, grayscale=False,
             loop=False, turn_on_agg=True):
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot temporaries for making gif
    # use random code to try and avoid deleting surprise files...
    random_code = random.randrange(2 ** 32)
    pre = str(random_code)
    for n, arr_i in enumerate(arr):
        plt.matshow(arr_i.reshape(plot_width, plot_height), cmap="gray")
        plt.axis('off')
        if list_text_per_frame is not None:
            text = list_text_per_frame[n]
            if list_text_per_frame_color is not None:
                color = list_text_per_frame_color[n]
            else:
                color = "white"
            plt.text(0, plot_height, text, color=color,
                     fontsize=2 * plot_height)
        # This looks rediculous but should count the number of digit places
        # also protects against multiple runs
        # plus 1 is to maintain proper ordering
        plotpath = '__%s_giftmp_%s.png' % (str(n).zfill(len(
            str(len(arr))) + 1), pre)
        plt.savefig(plotpath)
        plt.close()

    # make gif
    assert delay >= 1
    gif_delay = int(delay)
    basestr = "convert __*giftmp_%s.png -delay %s " % (pre, str(gif_delay))
    if loop:
        basestr += "-loop 1 "
    else:
        basestr += "-loop 0 "
    if grayscale:
        basestr += "-depth 8 -type Grayscale -depth 8 "
    basestr += "-resize %sx%s " % (str(int(5 * plot_width)),
                                   str(int(5 * plot_height)))
    basestr += gif_name
    print("Attempting gif")
    print(basestr)
    subprocess.call(basestr, shell=True)
    filelist = glob.glob("__*giftmp_%s.png" % pre)
    for f in filelist:
        os.remove(f)


def concatenate(tensor_list, name, axis=0, force_cast_to_float=True):
    """
    Wrapper to `theano.tensor.concatenate`.
    """
    if force_cast_to_float:
        tensor_list = cast_to_float(tensor_list)
    out = tensor.concatenate(tensor_list, axis=axis)
    conc_dim = int(sum([calc_expected_dim(inp)
                   for inp in tensor_list]))
    # This may be hosed... need to figure out how to generalize
    shape = list(expression_shape(tensor_list[0]))
    shape[axis] = conc_dim
    new_shape = tuple(shape)
    tag_expression(out, name, new_shape)
    return out


def theano_repeat(arr, n_repeat, stretch=False):
    """ Create repeats of 2D array using broadcasting. """
    if arr.dtype not in ["float32", "float64"]:
        arr = tensor.cast(arr, "int32")
    if stretch:
        arg1 = arr.dimshuffle((0, 'x', 1))
        arg2 = tensor.alloc(1., 1, n_repeat, arr.shape[1])
        arg2 = tensor.cast(arg2, arr.dtype)
        cloned = (arg1 * arg2).reshape((n_repeat * arr.shape[0], arr.shape[1]))
    else:
        arg1 = arr.dimshuffle(('x', 0, 1))
        arg2 = tensor.alloc(1., n_repeat, 1, arr.shape[1])
        arg2 = tensor.cast(arg2, arr.dtype)
        cloned = (arg1 * arg2).reshape((n_repeat * arr.shape[0], arr.shape[1]))
    shape = expression_shape(arr)
    name = expression_name(arr)
    # Stretched shapes are *WRONG*
    tag_expression(cloned, name + "_stretched", (shape[0], shape[1]))
    return cloned


def cast_to_float(list_of_inputs):
    # preserve name and shape info after cast
    input_names = [inp.name for inp in list_of_inputs]
    cast_inputs = [tensor.cast(inp, theano.config.floatX)
                   for inp in list_of_inputs]
    for n, inp in enumerate(cast_inputs):
        cast_inputs[n].name = input_names[n]
    return cast_inputs


def interpolate_between_points(arr, n_steps=50):
    assert len(arr) > 2
    assert n_steps > 1
    path = [path_between_points(start, stop, n_steps=n_steps)
            for start, stop in zip(arr[:-1], arr[1:])]
    path = np.vstack(path)
    return path


def path_between_points(start, stop, n_steps=100, dtype=theano.config.floatX):
    assert n_steps > 1
    step_vector = 1. / (n_steps - 1) * (stop - start)
    steps = np.arange(0, n_steps)[:, None] * np.ones((n_steps, len(stop)))
    steps = steps * step_vector + start
    return steps.astype(dtype)


def minibatch_indices(itr, minibatch_size):
    minibatch_indices = np.arange(0, len(itr), minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [len(itr)])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)


def convert_to_one_hot(itr, n_classes, dtype="int32"):
    is_three_d = False
    if type(itr) is np.ndarray:
        if len(itr.shape) == 3:
            is_three_d = True
    elif not isinstance(itr[0], numbers.Real):
        # Assume 3D list of list of list
        # iterable of iterable of iterable, feature dim must be consistent
        is_three_d = True

    if is_three_d:
        lengths = [len(i) for i in itr]
        one_hot = np.zeros((max(lengths), len(itr), n_classes), dtype=dtype)
        for n in range(len(itr)):
            one_hot[np.arange(lengths[n]), n, itr[n]] = 1
    else:
        one_hot = np.zeros((len(itr), n_classes), dtype=dtype)
        one_hot[np.arange(len(itr)), itr] = 1
    return one_hot


def save_checkpoint(save_path, items_dict):
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        pickle.dump(items_dict, f)
    sys.setrecursionlimit(old_recursion_limit)


def load_checkpoint(save_path):
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="rb") as f:
        items_dict = pickle.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return items_dict


def print_status_func(epoch_results):
    n_epochs_seen = len(epoch_results.values()[0])
    last_results = {k: v[-1] for k, v in epoch_results.items()}
    print("Epoch %i: %s" % (n_epochs_seen, last_results))


def print_and_checkpoint_status_func(save_path, checkpoint_dict, epoch_results):
    checkpoint_dict["previous_epoch_results"] = epoch_results
    save_checkpoint(save_path, checkpoint_dict)
    print_status_func(epoch_results)


def make_minibatch(arg, start, stop):
    """ Does not handle off-size minibatches """
    return [arg[start:stop]]


def gen_text_minibatch_func(one_hot_size):
    def apply(arg, start, stop):
        sli = arg[start:stop]
        expanded = convert_to_one_hot(sli, one_hot_size)
        lengths = [len(s) for s in sli]
        mask = np.zeros((max(lengths), len(sli)), dtype=theano.config.floatX)
        for n, l in enumerate(lengths):
            mask[np.arange(l), n] = 1.
        return expanded, mask
    return apply


def iterate_function(func, list_of_args, minibatch_size,
                     list_of_non_minibatch_args=None,
                     list_of_minibatch_functions=[make_minibatch],
                     list_of_output_names=None,
                     n_epochs=1000, n_status=50, status_func=None,
                     previous_epoch_results=None,
                     shuffle=False, random_state=None):
    """
    Minibatch args should come first
    If list_of_minbatch_functions is length 1, will be replicated to length of
    list_of_args.

    By far the craziest function in this library.
    """
    if previous_epoch_results is None:
        epoch_results = defaultdict(list)
    else:
        epoch_results = previous_epoch_results
    # Input checking and setup
    if shuffle:
        assert random_state is not None
    status_points = list(range(n_epochs))
    status_points = status_points[::n_epochs // n_status] + [status_points[-1]]

    for arg in list_of_args:
        assert len(arg) == len(list_of_args[0])

    indices = minibatch_indices(list_of_args[0], minibatch_size)
    if len(list_of_args[0]) % minibatch_size != 0:
        print ("length of dataset should be evenly divisible by "
               "minibatch_size.")
    if len(list_of_minibatch_functions) == 1:
        list_of_minibatch_functions = list_of_minibatch_functions * len(
            list_of_args)
    else:
        assert len(list_of_minibatch_functions) == len(list_of_args)
    # Function loop
    for e in range(n_epochs):
        results = defaultdict(list)
        if shuffle:
            random_state.shuffle(indices)
        for i, j in indices:
            minibatch_args = []
            for n, arg in enumerate(list_of_args):
                minibatch_args += list_of_minibatch_functions[n](arg, i, j)
            if list_of_non_minibatch_args is not None:
                all_args = minibatch_args + list_of_non_minibatch_args
            else:
                all_args = minibatch_args
            minibatch_results = func(*all_args)
            if type(minibatch_results) is not list:
                minibatch_results = [minibatch_results]
            for n, k in enumerate(minibatch_results):
                if list_of_output_names is not None:
                    assert len(list_of_output_names) == len(minibatch_results)
                    results[list_of_output_names[n]].append(
                        minibatch_results[n])
                else:
                    results[n].append(minibatch_results[n])
        avg_output = {r: np.mean(results[r]) for r in results.keys()}
        for k in avg_output.keys():
            epoch_results[k].append(avg_output[k])
        if e in status_points:
            if status_func is not None:
                epoch_number = e
                status_number = np.searchsorted(status_points, e)
                status_func(status_number, epoch_number, epoch_results)
    return epoch_results


def as_shared(arr, name=None):
    """ Quick wrapper for theano.shared """
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def np_zeros(shape):
    """ Builds a numpy variable filled with zeros """
    return np.zeros(shape).astype(theano.config.floatX)


def np_rand(shape, random_state):
    # Make sure bounds aren't the same
    return random_state.uniform(low=-0.08, high=0.08, size=shape).astype(
        theano.config.floatX)


def np_randn(shape, random_state):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * random_state.randn(*shape)).astype(theano.config.floatX)


def np_tanh_fan(shape, random_state):
    # The . after the 6 is critical! shape has dtype int...
    bound = np.sqrt(6. / np.sum(shape))
    return random_state.uniform(low=-bound, high=bound,
                                size=shape).astype(theano.config.floatX)


def np_sigmoid_fan(shape, random_state):
    return 4 * np_tanh_fan(shape, random_state)


def np_ortho(shape, random_state):
    """ Builds a theano variable filled with orthonormal random values """
    g = random_state.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


def names_in_graph(list_of_names, graph):
    """ Return true if all names are in the graph """
    return all([name in graph.keys() for name in list_of_names])


def add_arrays_to_graph(list_of_arrays, list_of_names, graph, strict=True):
    assert len(list_of_arrays) == len(list_of_names)
    arrays_added = []
    for array, name in zip(list_of_arrays, list_of_names):
        if name in graph.keys() and strict:
            raise ValueError("Name %s already found in graph!" % name)
        shared_array = as_shared(array, name=name)
        graph[name] = shared_array
        arrays_added.append(shared_array)


def make_shapename(name, shape):
    if len(shape) == 1:
        # vector, primarily init hidden state for RNN
        return name + "_kdl_" + str(shape[0]) + "x"
    else:
        return name + "_kdl_" + "x".join(map(str, list(shape)))


def parse_shapename(shapename):
    try:
        # Bracket for scan
        shape = shapename.split("_kdl_")[1].split("[")[0].split("x")
    except AttributeError:
        raise AttributeError("Unable to parse shapename. Has the expression "
                             "been tagged with a shape by tag_expression? "
                             " input shapename was %s" % shapename)
    if "[" in shapename.split("_kdl_")[1]:
        # inside scan
        shape = shape[1:]
    name = shapename.split("_kdl_")[0]
    # More cleaning to handle scan
    shape = tuple([int(s) for s in shape if s != ''])
    return name, shape


def add_datasets_to_graph(list_of_datasets, list_of_names, graph, strict=True,
                          list_of_test_values=None):
    assert len(list_of_datasets) == len(list_of_names)
    datasets_added = []
    for n, (dataset, name) in enumerate(zip(list_of_datasets, list_of_names)):
        if dataset.dtype != "int32":
            if len(dataset.shape) == 1:
                sym = tensor.vector()
            elif len(dataset.shape) == 2:
                sym = tensor.matrix()
            elif len(dataset.shape) == 3:
                sym = tensor.tensor3()
            else:
                raise ValueError("dataset %s has unsupported shape" % name)
        elif dataset.dtype == "int32":
            if len(dataset.shape) == 1:
                sym = tensor.ivector()
            elif len(dataset.shape) == 2:
                sym = tensor.imatrix()
            elif len(dataset.shape) == 3:
                sym = tensor.itensor3()
            else:
                raise ValueError("dataset %s has unsupported shape" % name)
        else:
            raise ValueError("dataset %s has unsupported dtype %s" % (
                name, dataset.dtype))
        if list_of_test_values is not None:
            sym.tag.test_value = list_of_test_values[n]
        tag_expression(sym, name, dataset.shape)
        datasets_added.append(sym)
    graph["__datasets_added__"] = datasets_added
    return datasets_added


def tag_expression(expression, name, shape):
    expression.name = make_shapename(name, shape)


def expression_name(expression):
    return parse_shapename(expression.name)[0]


def expression_shape(expression):
    return parse_shapename(expression.name)[1]


def calc_expected_dim(expression):
    # super intertwined with add_datasets_to_graph
    # Expect variables representing datasets in graph!!!
    # Function graph madness
    # Shape format is HxWxZ
    shape = expression_shape(expression)
    dim = shape[-1]
    return dim


def fetch_from_graph(list_of_names, graph):
    """ Returns a list of shared variables from the graph """
    if "__datasets_added__" not in graph.keys():
        # Check for dataset in graph
        raise AttributeError("No dataset in graph! Make sure to add "
                             "the dataset using add_datasets_to_graph")
    return [graph[name] for name in list_of_names]


def get_params_and_grads(graph, cost):
    grads = []
    params = []
    for k, p in graph.items():
        if k[:2] == "__":
            # skip private tags
            continue
        print("Computing grad w.r.t %s" % k)
        grad = tensor.grad(cost, p)
        params.append(p)
        grads.append(grad)
    return params, grads


def binary_crossentropy_nll(predicted_values, true_values):
    """ Returns likelihood compared to binary true_values """
    return (-true_values * tensor.log(predicted_values) - (
        1 - true_values) * tensor.log(1 - predicted_values)).sum(axis=-1)


def binary_entropy(values):
    return (-values * tensor.log(values)).sum(axis=-1)


def categorical_crossentropy_nll(predicted_values, true_values):
    """ Returns likelihood compared to one hot category labels """
    indices = tensor.argmax(true_values, axis=-1)
    rows = tensor.arange(true_values.shape[0])
    if predicted_values.ndim < 3:
        return -tensor.log(predicted_values)[rows, indices]
    elif predicted_values.ndim == 3:
        d0 = true_values.shape[0]
        d1 = true_values.shape[1]
        pred = predicted_values.reshape((d0 * d1, -1))
        ind = indices.reshape((d0 * d1,))
        s = tensor.arange(pred.shape[0])
        correct = -tensor.log(pred)[s, ind]
        return correct.reshape((d0, d1,))
    else:
        raise AttributeError("Tensor dim not supported")


def abs_error_nll(predicted_values, true_values):
    return tensor.abs_(predicted_values - true_values).sum(axis=-1)


def squared_error_nll(predicted_values, true_values):
    return tensor.sqr(predicted_values - true_values).sum(axis=-1)


def masked_cost(cost, mask):
    return cost * mask


def softplus(X):
    return tensor.nnet.softplus(X) + 1E-4


def relu(X):
    return X * (X > 1)


def softmax(X):
    # should work for both 2D and 3D
    e_X = tensor.exp(X - X.max(axis=-1, keepdims=True))
    out = e_X / e_X.sum(axis=-1, keepdims=True)
    return out


def linear(X):
    return X


def projection_layer(list_of_inputs, graph, name, proj_dim=None,
                     random_state=None, strict=True, init_func=np_tanh_fan,
                     func=linear):
    W_name = name + '_W'
    b_name = name + '_b'
    list_of_names = [W_name, b_name]
    if not names_in_graph(list_of_names, graph):
        assert proj_dim is not None
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        np_W = init_func((conc_input_dim, proj_dim), random_state)
        np_b = np_zeros((proj_dim,))
        add_arrays_to_graph([np_W, np_b], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    W, b = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, name, axis=-1)
    output = tensor.dot(conc_input, W) + b
    if func is not None:
        final = func(output)
    else:
        final = output
    shape = list(expression_shape(conc_input))
    # Projection is on last axis
    shape[-1] = proj_dim
    new_shape = tuple(shape)
    tag_expression(final, name, new_shape)
    return final


def linear_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                 strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=linear)


def sigmoid_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                  strict=True, init_func=np_sigmoid_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.nnet.sigmoid)


def tanh_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
               strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.nnet.tanh)


def softplus_layer(list_of_inputs, graph, name, proj_dim=None,
                   random_state=None, strict=True,
                   init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=softplus)


def exp_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
              strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.exp)


def relu_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
               strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=relu)


def softmax_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                  strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=softmax)


def softmax_sample_layer(list_of_multinomial_inputs, name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_multinomial = concatenate(list_of_multinomial_inputs, name, axis=1)
    shape = expression_shape(conc_multinomial)
    conc_multinomial /= len(list_of_multinomial_inputs)
    tag_expression(conc_multinomial, name, shape)
    samp = theano_rng.multinomial(pvals=conc_multinomial,
                                  dtype="int32")
    tag_expression(samp, name, (shape[0], shape[1]))
    return samp


def gaussian_sample_layer(list_of_mu_inputs, list_of_sigma_inputs,
                          name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, name, axis=1)
    conc_sigma = concatenate(list_of_sigma_inputs, name, axis=1)
    e = theano_rng.normal(size=(conc_sigma.shape[0],
                                conc_sigma.shape[1]),
                          dtype=conc_sigma.dtype)
    samp = conc_mu + conc_sigma * e
    shape = expression_shape(conc_sigma)
    tag_expression(samp, name, shape)
    return samp


def gaussian_log_sample_layer(list_of_mu_inputs, list_of_log_sigma_inputs,
                              name, random_state=None):
    """ log_sigma_inputs should be from a linear_layer """
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, name, axis=1)
    conc_log_sigma = concatenate(list_of_log_sigma_inputs, name, axis=1)
    e = theano_rng.normal(size=(conc_log_sigma.shape[0],
                                conc_log_sigma.shape[1]),
                          dtype=conc_log_sigma.dtype)

    samp = conc_mu + tensor.exp(0.5 * conc_log_sigma) * e
    shape = expression_shape(conc_log_sigma)
    tag_expression(samp, name, shape)
    return samp


def gaussian_kl(list_of_mu_inputs, list_of_sigma_inputs, name):
    conc_mu = concatenate(list_of_mu_inputs, name)
    conc_sigma = concatenate(list_of_sigma_inputs, name)
    kl = 0.5 * tensor.sum(-2 * tensor.log(conc_sigma) + conc_mu ** 2
                          + conc_sigma ** 2 - 1, axis=1)
    return kl


def gaussian_log_kl(list_of_mu_inputs, list_of_log_sigma_inputs, name):
    """ log_sigma_inputs should come from linear layer"""
    conc_mu = concatenate(list_of_mu_inputs, name)
    conc_log_sigma = 0.5 * concatenate(list_of_log_sigma_inputs, name)
    kl = 0.5 * tensor.sum(-2 * conc_log_sigma + conc_mu ** 2
                          + tensor.exp(conc_log_sigma) ** 2 - 1, axis=1)
    return kl


def switch_wrap(switch_func, if_true_var, if_false_var, name):
    switched = tensor.switch(switch_func, if_true_var, if_false_var)
    shape = expression_shape(if_true_var)
    assert shape == expression_shape(if_false_var)
    tag_expression(switched, name, shape)
    return switched


def rnn_scan_wrap(func, sequences=None, outputs_info=None, non_sequences=None,
                  n_steps=None, truncate_gradient=-1, go_backwards=False,
                  mode=None,
                  name=None, profile=False, allow_gc=None, strict=False):
    """ Expects 3D input sequences, dim 0 being the axis of iteration """
    # assumes 0th output of func is hidden state
    # necessary so that values out of scan can be tagged... ugh
    # shape_of_variables eliminates the need for this
    outputs, updates = theano.scan(func, sequences=sequences,
                                   outputs_info=outputs_info,
                                   non_sequences=non_sequences, n_steps=n_steps,
                                   truncate_gradient=truncate_gradient,
                                   go_backwards=go_backwards, mode=mode,
                                   name=name, profile=profile,
                                   allow_gc=allow_gc, strict=strict)
    s = expression_shape(sequences[0])
    shape_0 = s[0]
    if type(outputs) is list:
        for n, o in enumerate(outputs):
            s = expression_shape(outputs_info[n])
            # all sequences should be the same length
            shape_1 = s
            shape = (shape_0,) + shape_1
            tag_expression(outputs[n], name + "_%s_" % n, shape)
    else:
        s = expression_shape(outputs_info[0])
        shape_1 = s
        # combine tuples
        shape = (shape_0,) + shape_1
        tag_expression(outputs, name, shape)
    return outputs, updates


def tanh_recurrent_layer(list_of_inputs, list_of_hiddens, graph, name,
                         random_state=None, strict=True):
    # All inputs are assumed 2D as are hiddens
    # Everything is dictated by the size of the hiddens
    W_name = name + '_tanhrec_W'
    b_name = name + '_tanhrec_b'
    U_name = name + '_tanhrec_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        conc_hidden_dim = int(sum([calc_expected_dim(hid)
                                   for hid in list_of_hiddens]))
        shape = (conc_input_dim, conc_hidden_dim)
        np_W = np_rand(shape, random_state)
        np_b = np_zeros((shape[-1],))
        np_U = np_ortho((shape[-1], shape[-1]), random_state)
        add_arrays_to_graph([np_W, np_b, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    W, b, U = fetch_from_graph(list_of_names, graph)
    # per timestep
    conc_input = concatenate(list_of_inputs, name + "_input", axis=-1)
    conc_hidden = concatenate(list_of_hiddens, name + "_hidden", axis=-1)
    output = tensor.tanh(tensor.dot(conc_input, W) + b +
                         tensor.dot(conc_hidden, U))
    # remember this is number of dims per timestep!
    shape = expression_shape(conc_input)
    tag_expression(output, name, shape)
    return output


def easy_tanh_recurrent(list_of_inputs, mask, hidden_dim, graph, name,
                        random_state,
                        one_step=False):
    # an easy interface to lstm recurrent nets
    shape = expression_shape(list_of_inputs[0])
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # an easy interface to tanh recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))

    def step(x_t, m_t, h_tm1):
        h_ti = tanh_recurrent_layer([x_t], [h_tm1], graph,
                                    name + '_easy_tanh_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    if one_step:
        conc_input = concatenate(list_of_inputs, name + "_easy_tanh_step",
                                 axis=-1)
        shape = expression_shape(conc_input)
        sliced = conc_input[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h = step(sliced, h0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        h, updates = rnn_scan_wrap(step, name=name + '_easy_tanh_scan',
                                   sequences=list_of_inputs + [mask],
                                   outputs_info=[h0_sym])
    return h


def gru_recurrent_layer(list_of_inputs, list_of_hiddens, graph, name,
                        random_state=None, strict=True):
    W_name = name + '_grurec_W'
    b_name = name + '_grurec_b'
    U_name = name + '_grurec_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        conc_hidden_dim = int(sum([calc_expected_dim(hid)
                                   for hid in list_of_hiddens]))
        shape = (conc_input_dim, conc_hidden_dim)
        np_W = np.hstack([np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state)])
        np_b = np_zeros((3 * shape[1],))
        np_U = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state)])
        add_arrays_to_graph([np_W, np_b, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        dim = shape[1]
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]
    W, b, U = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, name + "_input", axis=0)
    conc_hidden = concatenate(list_of_hiddens, name + "_hidden", axis=0)
    proj_i = tensor.dot(conc_input, W) + b
    proj_h = tensor.dot(conc_hidden, U)
    r = tensor.nnet.sigmoid(_slice(proj_i, 1)
                            + _slice(proj_h, 1))
    z = tensor.nnet.sigmoid(_slice(proj_i, 2)
                            + _slice(proj_h, 2))
    candidate_h = tensor.tanh(_slice(proj_i, 0) + r * _slice(proj_h, 0))
    output = z * conc_hidden + (1. - z) * candidate_h
    # remember this is number of dims per timestep!
    tag_expression(output, name, (shape[1],))
    return output


def easy_gru_recurrent(list_of_inputs, mask, hidden_dim, graph, name,
                       random_state, one_step=False):
    # an easy interface to lstm recurrent nets
    shape = expression_shape(list_of_inputs[0])
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # an easy interface to tanh recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))

    def step(x_t, m_t, h_tm1):
        h_ti = gru_recurrent_layer([x_t], [h_tm1], graph,
                                   name + '_easy_gru_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    if one_step:
        conc_input = concatenate(list_of_inputs, name + "_easy_gru_step",
                                 axis=-1)
        shape = expression_shape(conc_input)
        sliced = conc_input[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h = step(sliced, h0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        h, updates = rnn_scan_wrap(step, name=name + '_easy_gru_scan',
                                   sequences=list_of_inputs + [mask],
                                   outputs_info=[h0_sym])
    return h


def lstm_recurrent_layer(list_of_inputs, list_of_hiddens, list_of_cells,
                         graph, name, random_state=None, strict=True):
    W_name = name + '_lstmrec_W'
    b_name = name + '_lstmrec_b'
    U_name = name + '_lstmrec_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        conc_hidden_dim = int(sum([calc_expected_dim(hid)
                                   for hid in list_of_hiddens]))
        conc_cell_dim = int(sum([calc_expected_dim(hid)
                                 for hid in list_of_cells]))
        assert conc_hidden_dim == conc_cell_dim
        shape = (conc_input_dim, conc_hidden_dim)
        np_W = np.hstack([np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state)])
        np_b = np_zeros((4 * shape[1],))
        np_U = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state)])
        add_arrays_to_graph([np_W, np_b, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        dim = shape[1]
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]
    W, b, U = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, name + "_input", axis=0)
    conc_hidden = concatenate(list_of_hiddens, name + "_hidden", axis=0)
    conc_cell = concatenate(list_of_cells, name + "_cell", axis=0)
    proj_i = tensor.dot(conc_input, W) + b
    proj_h = tensor.dot(conc_hidden, U)
    # input output forget and cell gates
    ig = tensor.nnet.sigmoid(_slice(proj_i, 0) + _slice(proj_h, 0))
    fg = tensor.nnet.sigmoid(_slice(proj_i, 1) + _slice(proj_h, 1))
    og = tensor.nnet.sigmoid(_slice(proj_i, 2) + _slice(proj_h, 2))
    cg = tensor.tanh(_slice(proj_i, 3) + _slice(proj_h, 3))
    c = fg * conc_cell + ig * cg
    h = og * tensor.tanh(c)
    tag_expression(h, name + "_hidden", (shape[1],))
    tag_expression(c, name + "_cell", (shape[1],))
    return h, c


def easy_lstm_recurrent(list_of_inputs, mask, hidden_dim, graph, name,
                        random_state, one_step=False):
    # an easy interface to lstm recurrent nets
    shape = expression_shape(list_of_inputs[0])
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # an easy interface to tanh recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))

    c0 = np_zeros((shape[1], hidden_dim))
    c0_sym = as_shared(c0, name)
    tag_expression(c0_sym, name, (shape[1], hidden_dim))

    def step(x_t, m_t, h_tm1, c_tm1):
        h_ti, c_ti = lstm_recurrent_layer([x_t], [h_tm1], [c_tm1], graph,
                                          name + '_easy_lstm_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        c_t = m_t[:, None] * c_ti + (1 - m_t)[:, None] * c_tm1
        return h_t, c_t

    if one_step:
        conc_input = concatenate(list_of_inputs, name + "_easy_lstm_step",
                                 axis=-1)
        shape = expression_shape(conc_input)
        sliced = conc_input[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h, c = step(sliced, h0_sym, c0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        [h, c], updates = rnn_scan_wrap(step, name=name + '_easy_lstm_scan',
                                        sequences=list_of_inputs + [mask],
                                        outputs_info=[h0_sym, c0_sym])
    return h


if __name__ == "__main__":
    # Sample usages
    # random state so script is deterministic
    random_state = np.random.RandomState(1999)
    # home of the computational graph
    graph = OrderedDict()

    # input (where first dimension is time)
    X_sym = tensor.dmatrix()
    # target (where first dimension is time)
    y_sym = tensor.dmatrix()
    # initial hidden state of the RNN
    h0 = tensor.dvector()

    # Parameters of the model
    cost = binary_crossentropy_nll(X_sym, X_sym)
    params = list(graph.values())
    grads = tensor.grad(cost, params)
    # Use stochastic gradient descent to optimize
    opt = sgd(params)
    learning_rate = 0.001
    updates = opt.updates(params, grads, learning_rate)
