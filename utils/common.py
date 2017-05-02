from sklearn.metrics import fbeta_score
import numpy as np

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def iterate_minibatches(inputs, batchsize=10):

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt]

    if len(inputs) % batchsize != 0:
        yield np.array(inputs)[- (len(inputs) % batchsize):]


def parallel_shuffle(X, y, shuffle=True):
    if shuffle:
        count = len(X)
        indices = range(count)
        np.random.shuffle(indices)
        return [X[indices], y[indices]]

    return None

