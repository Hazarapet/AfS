from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def model(weights_path=None):
    inputs = Input(shape=(4, 128, 128))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(bn1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    bn2 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(bn2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv4)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=1)
    dr1 = Dropout(0.2)(up6)
    conv5 = Conv2D(128, (3, 3), activation='elu', padding='same')(dr1)
    conv5 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv5)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=1)
    dr2 = Dropout(0.2)(up7)
    conv6 = Conv2D(64, (3, 3), activation='elu', padding='same')(dr2)
    conv6 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv6)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv6), bn2], axis=1)
    dr3 = Dropout(0.2)(up8)
    conv7 = Conv2D(32, (3, 3), activation='elu', padding='same')(dr3)
    conv7 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv7)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    _m = Model(inputs=[inputs], outputs=[conv10])

    _model = Sequential()
    _model.add(_m)

    _model.add(Conv2D(32, (3, 3)))
    _model.add(Activation('relu'))

    _model.add(Conv2D(32, (3, 3)))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(Conv2D(64, (3, 3)))
    _model.add(Activation('relu'))

    _model.add(Conv2D(64, (3, 3)))
    _model.add(Activation('relu'))

    _model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    _model.add(Flatten())

    _model.add(Dense(128, kernel_regularizer=l2(1e-4)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))

    _model.add(Dense(1, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/water/structures/']
