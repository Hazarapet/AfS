import h5py
import numpy as np
from keras import layers
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    _model = Sequential()
    _model.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128)))
    _model.add(Conv2D(64, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(64, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(256, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(256, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(256, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dense layers
    _model.add(Flatten())

    _model.add(Dense(512, kernel_regularizer=l2(1e-5)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.1))

    _model.add(Dense(512, kernel_regularizer=l2(1e-5)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.1))

    _model.add(Dense(17, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/main/structures/']


def load_weights(_model, weights_path):
    f = h5py.File(weights_path, "r")

    weights = {k: v for k, v in zip(f.keys(), f.values())}

    for layer in _model.layers:

        if layer.name == 'zero_padding2d_10':
            break

        w = weights[layer.name]
        ar = []
        for v in w.values():
            for vv in v.values():
                ar.append(vv[()])

        # ar = np.array(ar).astype(np.float32)
        # layer.set_weights(ar)

        print 'layer "' + layer.name + '" weights are loaded'

    return _model
