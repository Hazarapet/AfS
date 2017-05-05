import keras
import numpy as np
import keras.backend as K

DISTRIBUTION_TAILOR = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).astype(np.int8)

def reg_binary_cross_entropy(l=1, p=.5):
    def func(y_true, y_pred):  # 2D tensor variable
        _selected = DISTRIBUTION_TAILOR * y_pred  # (1, 3) * (batch, 3)
        _excessive = (_selected > p) * p
        _tailor = K.mean(_selected - _excessive)  # > 0
        return keras.losses.binary_crossentropy(y_true, y_pred) + l*_tailor

    return func
