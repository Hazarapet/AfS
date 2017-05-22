import keras
import common
import numpy as np
import keras.backend as K

def f2_binary_cross_entropy(l=1):
    def func(y_true, y_pred):  # 2D tensor variable
        return keras.losses.binary_crossentropy(y_true, y_pred) - l * K.log(common.f2_score(y_true, y_pred))

    return func
