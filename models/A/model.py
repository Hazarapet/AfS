from keras.models import Sequential
from keras.layers import merge
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def model(weights_path=None):
    _regular_input = Sequential()
    _regular_input.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128), name='regular_input'))

    _sobel_input = Sequential()
    _sobel_input.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128), name='sobel_input'))

    _model = Sequential()
    _model.add(merge([_regular_input, _sobel_input], mode='concat'))

    _model.add(Conv2D(64, (3, 3)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(64, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(128, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(256, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(256, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(256, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(ZeroPadding2D((1, 1)))
    _model.add(Conv2D(512, (3, 3)))
    # _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dense layers
    _model.add(Flatten())

    _model.add(Dense(512, kernel_regularizer=l2(1e-4)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))

    _model.add(Dense(256, kernel_regularizer=l2(1e-4)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))

    _model.add(Dense(17, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/A/structures/']
