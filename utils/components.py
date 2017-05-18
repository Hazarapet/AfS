import keras
import numpy as np
import keras.backend as K

DISTRIBUTION_TAILOR = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).astype(np.int8)

def reg_binary_cross_entropy(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)
