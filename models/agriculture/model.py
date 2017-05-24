from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    # red, green, blue, ndwi, ndvi, ior, gemi
    _model = Sequential()
    _model.add(ZeroPadding2D((1, 1), input_shape=(7, 128, 128)))
    _model.add(Conv2D(64, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(64, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    _model.add(BatchNormalization(axis=1))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

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

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    _model.add(Flatten())

    _model.add(Dense(128, kernel_regularizer=l2(1e-5)))
    _model.add(Activation('relu'))

    _model.add(Dense(64, kernel_regularizer=l2(1e-5)))
    _model.add(Activation('relu'))

    _model.add(Dense(1, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/agriculture/structures/']
