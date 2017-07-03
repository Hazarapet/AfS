import h5py
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Input, concatenate
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    _input = Input((3, 224, 224))
    _m = ResNet50(weights=None, include_top=False, input_tensor=_input, input_shape=(3, 224, 224))
    _m.load_weights('models/nm/structures/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')

    for layer in _m.layers:
        layer.trainable = False

    x = _m.output
    x = Dense(512, name='my_dense_1')(x)
    # x = BatchNormalization(axis=1, name='my_bn_1')(x)
    x = Activation('relu', name='my_act_1')(x)

    x = Dense(17, name='my_dense_2')(x)
    x = Activation('sigmoid', name='my_output')(x)

    _model = Model(inputs=_m.input, outputs=x)

    return [_model, 'models/nm/structures/']