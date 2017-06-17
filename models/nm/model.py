from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def conv_block(input, nm_filter):
    conv11 = Conv2D(nm_filter, (1, 1), padding='same', use_bias=False)(input)
    conv33 = Conv2D(nm_filter, (3, 3), padding='same', use_bias=False)(conv11)
    bn = BatchNormalization(axis=1)(conv33)
    act = Activation('relu')(bn)

    return act

def model(weights_path=None):
    k = 12
    nm_filter = 64
    input = Input((3, 224, 224))

    # ------------------------------------------------------
    start_conv = Conv2D(nm_filter, (7, 7), strides=(2, 2), padding='same', name='gateway_conv')(input)
    start_bn = BatchNormalization(axis=1, name='gateway_bn')(start_conv)
    start_act = Activation('relu', name='gateway_act')(start_bn)

    start_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='gateway_max_pool')(start_act)

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    # ---------------------- 112x112 -------------------------
    tmp_input = start_pool

    for i in range(6):
        if i > 0:
            tmp_input = concatenate([tmp_input, conv1], axis=1)
        conv1 = conv_block(tmp_input, nm_filter)
        nm_filter += k

    # -----------------------------------------------------
    # --------------------- Bridge 1 ----------------------
    bridge_conv11 = Conv2D(nm_filter, (3, 3), padding='same')(conv1)
    bridge_bn11 = BatchNormalization(axis=1)(bridge_conv11)
    bridge_act11 = Activation('relu')(bridge_bn11)

    bridge_pool11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act11)

    # ------------------------------------------------------
    # ------------------ Conv Block 2 ----------------------
    # ---------------------- 32x32 -------------------------
    tmp_input = bridge_pool11
    for i in range(12):
        if i > 0:
            tmp_input = concatenate([tmp_input, conv2], axis=1)
        conv2 = conv_block(tmp_input, nm_filter)
        nm_filter += k
    # -----------------------------------------------------
    # --------------------- Bridge 2 ----------------------
    bridge_conv21 = Conv2D(nm_filter, (3, 3), padding='same')(conv2)
    bridge_bn21 = BatchNormalization(axis=1)(bridge_conv21)
    bridge_act21 = Activation('relu')(bridge_bn21)

    bridge_pool21 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act21)

    # ------------------------------------------------------
    # ------------------ Conv Block 3 ----------------------
    # ---------------------- 16x16 -------------------------
    tmp_input = bridge_pool21
    for i in range(10):
        if i > 0:
            tmp_input = concatenate([tmp_input, conv3], axis=1)
        conv3 = conv_block(tmp_input, nm_filter)
        nm_filter += k
    # -----------------------------------------------------
    # --------------------- Bridge 3 ----------------------
    bridge_conv31 = Conv2D(nm_filter, (3, 3), padding='same')(conv3)
    bridge_bn31 = BatchNormalization(axis=1)(bridge_conv31)
    bridge_act31 = Activation('relu')(bridge_bn31)

    bridge_pool31 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act31)

    # ------------------------------------------------------
    # ------------------ Conv Block 4 ----------------------
    # ---------------------- 8x8 -------------------------
    tmp_input = bridge_pool31
    for i in range(10):
        if i > 0:
            tmp_input = concatenate([tmp_input, conv4], axis=1)
        conv4 = conv_block(tmp_input, nm_filter)
        nm_filter += k

    # -----------------------------------------------------
    # --------------------- Bridge 4 ----------------------
    bridge_conv41 = Conv2D(nm_filter, (3, 3), padding='same')(conv4)
    bridge_bn41 = BatchNormalization(axis=1)(bridge_conv41)
    bridge_act41 = Activation('relu')(bridge_bn41)

    bridge_pool41 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(bridge_act41)

    # Dense layers
    flt = Flatten()(bridge_pool41)

    dense1 = Dense(512, kernel_regularizer=l2(1e-5))(flt)
    # dnbn1 = BatchNormalization(axis=1)(dense1)
    dnact1 = Activation('relu')(dense1)
    dndrop1 = Dropout(0.2)(dnact1)

    output = Dense(17, activation='sigmoid')(dndrop1)

    _model = Model(inputs=input, outputs=output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
