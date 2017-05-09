from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def model(weights_path=None):
    inputs = Input(shape=(3, 128, 128))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    # bn3 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    # bn4 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    # bn5 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # bn6 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    # bn7 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    # bn8 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    # bn9 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    # bn10 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    dr1 = Dropout(0.4)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(dr1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    dr2 = Dropout(0.4)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(dr2)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    dr3 = Dropout(0.4)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(dr3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), bn2], axis=1)
    dr4 = Dropout(0.4)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(dr4)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='relu')(conv9)

    _m = Model(inputs=[inputs], outputs=[conv10])

    _model = Sequential()
    _model.add(_m)

    # Dense layers
    _model.add(Flatten())

    _model.add(Dense(256, kernel_regularizer=l2(1e-4)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))

    _model.add(Dense(17, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/UNET/structures/']
