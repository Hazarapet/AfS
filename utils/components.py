import keras
import keras.backend as K

DISTRIBUTION_TAILOR = [0, 1, 0]

def reg_binary_cross_entropy(l=1, p=.5):
    def func(y_true, y_pred): # 2D tensor variable
        _excessive = DISTRIBUTION_TAILOR * y_pred # (1, 3) * (batch, 3) #output = 3
        _excessive = (_excessive > p) * p
        _tailor = K.mean(y_pred - _excessive) # > 0
        return keras.objectives.binary_crossentropy(y_true, y_pred) + l*_tailor

    return func
