from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    _model = Sequential()
    _model.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128)))
    _model.add(Conv2D(32, (3, 3)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(32, (3, 3)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(64, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(64, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    _model.add(Flatten())

    _model.add(Dense(128, kernel_regularizer=l2(1e-4)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))

    _model.add(Dense(32, kernel_regularizer=l2(1e-4)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))

    _model.add(Dense(2, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/water/structures/']