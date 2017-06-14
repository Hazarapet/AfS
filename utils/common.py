import os
import shutil
import numpy as np
import keras.backend as K
from sklearn.metrics import fbeta_score


def f2_score_alt(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true).astype(np.float16), np.array(y_pred).astype(np.float16)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def f2_score(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def iterate_minibatches(inputs, batchsize=10):

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt]

    if len(inputs) % batchsize != 0:
        yield np.array(inputs)[- (len(inputs) % batchsize):]


def create_folder(dir_name=None):
    if os.path.exists(dir_name) and dir_name:
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)


def parallel_shuffle(x, y, shuffle=True):
    if shuffle:
        count = len(x)
        indices = range(count)
        np.random.shuffle(indices)
        return [[x[i] for i in indices], [y[i] for i in indices]]

    return [x, y]

def aug(array, input):
    rt90 = np.rot90(input, 1, axes=(1, 2))
    array.append(rt90)

    # flip h
    flip_h = np.flip(input, 2)
    array.append(flip_h)

    # flip v
    flip_v = np.flip(input, 1)
    array.append(flip_v)

    return array
