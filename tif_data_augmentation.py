import cv2
import numpy as np

def rotate90(array, axis=(0, 1)):
    return np.rot90(array, axes=axis)

def rotate180(array, axis=(0, 1)):
    return np.rot90(array, 2, axes=axis)

def flip_h(array):
    return np.fliplr(array)

def flip_v(array):
    return np.flipud(array)

def augment(example, rot90=False, rot180=False, f_v=False, f_h=False):
    augmented = []

    if rot90:
        augmented.append(rotate90(example))

    if rot180:
        augmented.append(rotate180(example))

    if f_h:
        augmented.append(flip_h(example))

    if f_v:
        augmented.append(flip_v(example))

    return augmented

def crop256_to_128(example):
    assert example.shape[0] > 256 and example.shape[1] > 256
    crops = []

    crops.append(example[:128, :128])
    crops.append(example[128:, :128])
    crops.append(example[:128, 128:])
    crops.append(example[128:, 128:])

    return crops