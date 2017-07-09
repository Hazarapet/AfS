import os
import cv2
import shutil
import numpy as np
import keras.backend as K
from sklearn.metrics import fbeta_score

thres = [0.17, 0.22, 0.19, 0.26, 0.25, 0.12, 0.23, 0.23, 0.17, 0.17, 0.25, 0.27, 0.34, 0.07, 0.16, 0.16, 0.24]


def f2_score_alt(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true).astype(np.float16), np.array(y_pred).astype(np.float16)

    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def f2_score(y_true, y_pred, threshold_shift=.3):
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


def mean_pred(y_pred):
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


def agg(array):
    return np.mean(array, axis=0)


def aug(array, input):
    # input's shape (cn, w, h)
    rn1 = np.random.randint(0, 32)
    rn2 = np.random.randint(input.shape[1] - 32, input.shape[1])

    # rotate 90
    rt90 = np.rot90(input, 1, axes=(1, 2))
    array.append(rt90)

    # flip h
    flip_h = np.flip(input, 2)
    array.append(flip_h)

    # flip v
    flip_v = np.flip(input, 1)
    array.append(flip_v)

    # rotate 90, flip v
    rot90_flip_v = np.rot90(flip_v, 1, axes=(1, 2))
    array.append(rot90_flip_v)

    # rotate 90, flip h
    rot90_flip_h = np.rot90(flip_h, 1, axes=(1, 2))
    array.append(rot90_flip_h)

    # random crop with 32px shift
    crop = input.transpose((1, 2, 0))
    crop = cv2.resize(crop[rn1:rn2, rn1:rn2], (crop.shape[0], crop.shape[1]))
    crop = crop.transpose((2, 0, 1))
    array.append(crop)

    # crop + rotate 90
    crop_rt90 = np.rot90(crop, 1, axes=(1, 2))
    array.append(crop_rt90)

    # crop + flip h
    crop_flip_h = np.flip(crop, 2)
    array.append(crop_flip_h)

    # crop + flip v
    crop_flip_v = np.flip(crop, 1)
    array.append(crop_flip_v)

    return array


def ensemble(array):
    new_array = []
    for cl in range(array.shape[1]):
        cn = list(array[:, cl]).count(1)
        all_cn = array.shape[0]
        if cn >= all_cn / 2.:
            new_array.append(1)
        else:
            new_array.append(0)

    return new_array


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = f2_score(y, p2).eval()
    return score

  x = [0.2] * 17
  best_sc = 0
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= float(resolution)
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

    best_sc = best_score

  return x, best_sc.tolist()