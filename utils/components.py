import keras
import common
import keras.backend as K


def f2_binary_cross_entropy(l=1):
    def func(y_true, y_pred):  # 2D tensor variable
        f2 = common.f2_score(y_true, y_pred, .2)
        # K.log(f2 + K.epsilon()) to avoid getting too big numbers if f2 == 0
        return keras.losses.binary_crossentropy(y_true, y_pred) - l * K.log(f2 + 100 * K.epsilon())

    return func
