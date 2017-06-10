from keras.models import Sequential, Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def model(weights_path=None):
    input = Input((8, 128, 128))

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    conv11 = Conv2D(32, (3, 3))(input)
    bn11 = BatchNormalization(axis=1)(conv11)
    act11 = Activation('relu')(bn11)

    conv12 = Conv2D(32, (3, 3))(act11)
    bn12 = BatchNormalization(axis=1)(conv12)
    act12 = Activation('relu')(bn12)

    conv13 = Conv2D(32, (3, 3))(act12)
    bn13 = BatchNormalization(axis=1)(conv13)
    act13 = Activation('relu')(bn13)

    conv14 = Conv2D(32, (3, 3))(act13)
    bn14 = BatchNormalization(axis=1)(conv14)
    act14 = Activation('relu')(bn14)

    pool14 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act14)

    # ------------------------------------------------------
    # ------------------ Conv Block 2 ----------------------
    prev1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input)
    prev_conv1 = Conv2D(8, (5, 5))(prev1)
    concat1 = concatenate([prev_conv1, pool14], axis=1)

    conv21 = Conv2D(64, (3, 3))(concat1)
    bn21 = BatchNormalization(axis=1)(conv21)
    act21 = Activation('relu')(bn21)

    conv22 = Conv2D(64, (3, 3))(act21)
    bn22 = BatchNormalization(axis=1)(conv22)
    act22 = Activation('relu')(bn22)

    conv23 = Conv2D(64, (3, 3))(act22)
    bn23 = BatchNormalization(axis=1)(conv23)
    act23 = Activation('relu')(bn23)

    conv24 = Conv2D(64, (3, 3))(act23)
    bn24 = BatchNormalization(axis=1)(conv24)
    act24 = Activation('relu')(bn24)

    pool24 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act24)

    # ------------------------------------------------------
    # ------------------ Conv Block 3 ----------------------
    prev2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pool14)
    prev_conv2 = Conv2D(8, (5, 5))(prev2)
    concat2 = concatenate([prev_conv2, pool24], axis=1)

    conv31 = Conv2D(128, (3, 3))(concat2)
    bn31 = BatchNormalization(axis=1)(conv31)
    act31 = Activation('relu')(bn31)

    conv32 = Conv2D(128, (3, 3))(act31)
    bn32 = BatchNormalization(axis=1)(conv32)
    act32 = Activation('relu')(bn32)

    conv33 = Conv2D(128, (3, 3))(act32)
    bn33 = BatchNormalization(axis=1)(conv33)
    act33 = Activation('relu')(bn33)

    conv34 = Conv2D(128, (3, 3))(act33)
    bn34 = BatchNormalization(axis=1)(conv34)
    act34 = Activation('relu')(bn34)

    pool34 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act34)

    # ------------------------------------------------------
    # ------------------ Conv Block 4 ----------------------
    prev3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pool24)
    prev_conv3 = Conv2D(8, (5, 5))(prev3)
    concat3 = concatenate([prev_conv3, pool34], axis=1)

    conv41 = Conv2D(256, (3, 3))(concat3)
    bn41 = BatchNormalization(axis=1)(conv41)
    act41 = Activation('relu')(bn41)

    conv42 = Conv2D(256, (3, 3))(act41)
    bn42 = BatchNormalization(axis=1)(conv42)
    act42 = Activation('relu')(bn42)

    conv43 = Conv2D(256, (3, 3))(act42)
    bn43 = BatchNormalization(axis=1)(conv43)
    act43 = Activation('relu')(bn43)

    conv44 = Conv2D(256, (3, 3))(act43)
    bn44 = BatchNormalization(axis=1)(conv44)
    act44 = Activation('relu')(bn44)

    pool44 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act44)

    # Dense layers
    _model = Sequential()
    _model.add(Model(input=input, outputs=pool44))

    _model.add(Flatten())

    _model.add(Dense(512, kernel_regularizer=l2(1e-5)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.1))

    _model.add(Dense(256, kernel_regularizer=l2(1e-5)))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.1))

    _model.add(Dense(17, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
