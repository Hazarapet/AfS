from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def conv_block(input, nm_filter, dp=0.1):
    bn11 = BatchNormalization(axis=1)(input)
    act11 = Activation('relu')(bn11)
    conv11 = Conv2D(nm_filter, (1, 1), padding='same', use_bias=False)(act11)

    bn33 = BatchNormalization(axis=1)(conv11)
    act33 = Activation('relu')(bn33)
    conv33 = Conv2D(nm_filter, (3, 3), padding='same', use_bias=False)(act33)

    drp = Dropout(dp)(conv33)

    return drp


def model(weights_path=None):
    k = 32
    nm_filter = 64
    input = Input((3, 224, 224))

    # ------------------------------------------------------
    start_conv = ZeroPadding2D((3, 3), name='gateway_padding3x3')(input)
    start_conv = Conv2D(2*k, (7, 7), strides=(2, 2), name='gateway_conv', use_bias=False)(start_conv)
    start_conv = BatchNormalization(axis=1, name='gateway_bn')(start_conv)
    start_conv = Activation('relu', name='gateway_act')(start_conv)

    start_conv = ZeroPadding2D((1, 1), name='gateway_padding1x1')(start_conv)
    start_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='gateway_max_pool')(start_conv)

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    # ---------------------- 56x56 -------------------------
    tmp_input = start_pool

    for i in range(6):
        conv1 = conv_block(tmp_input, nm_filter)
        tmp_input = concatenate([tmp_input, conv1], axis=1)

        nm_filter += k

    # -----------------------------------------------------
    # --------------------- Bridge 1 ----------------------

    bridge_bn11 = BatchNormalization(axis=1)(conv1)
    bridge_act11 = Activation('relu')(bridge_bn11)
    bridge_conv11 = Conv2D(int(nm_filter * 0.5), (1, 1), padding='same', use_bias=False)(bridge_act11)

    bridge_pool11 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_conv11)

    # ------------------------------------------------------
    # ------------------ Conv Block 2 ----------------------
    # ---------------------- 28x28 -------------------------

    tmp_input = bridge_pool11
    for i in range(6):
        conv2 = conv_block(tmp_input, nm_filter)
        tmp_input = concatenate([tmp_input, conv2], axis=1)

        nm_filter += k

    # -----------------------------------------------------
    # --------------------- Bridge 2 ----------------------
    bridge_bn21 = BatchNormalization(axis=1)(conv2)
    bridge_act21 = Activation('relu')(bridge_bn21)
    bridge_conv21 = Conv2D(int(nm_filter * 0.5), (1, 1), padding='same', use_bias=False)(bridge_act21)

    bridge_pool21 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_conv21)

    # ------------------------------------------------------
    # ------------------ Conv Block 3 ----------------------
    # ---------------------- 14x14 -------------------------

    tmp_input = bridge_pool21
    for i in range(6):
        conv3 = conv_block(tmp_input, nm_filter)
        tmp_input = concatenate([tmp_input, conv3], axis=1)

        nm_filter += k

    # -----------------------------------------------------
    # --------------------- Bridge 3 ----------------------
    bridge_bn31 = BatchNormalization(axis=1)(conv3)
    bridge_act31 = Activation('relu')(bridge_bn31)
    bridge_conv31 = Conv2D(int(nm_filter * 0.5), (1, 1), padding='same', use_bias=False)(bridge_act31)

    bridge_pool31 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_conv31)

    # ------------------------------------------------------
    # ------------------ Conv Block 4 ----------------------
    # ---------------------- 7x7 -------------------------

    tmp_input = bridge_pool31
    for i in range(6):
        conv4 = conv_block(tmp_input, nm_filter)
        tmp_input = concatenate([tmp_input, conv4], axis=1)

        nm_filter += k

    # -----------------------------------------------------
    # --------------------- Bridge 4 ----------------------
    bridge_bn41 = BatchNormalization(axis=1)(conv4)
    bridge_act41 = Activation('relu')(bridge_bn41)

    bridge_pool41 = GlobalAveragePooling2D(pool_size=(7, 7), strides=(1, 1))(bridge_act41)

    # Dense layers

    # dense1 = Dense(512, kernel_regularizer=l2(1e-5))(flt)
    # dnbn1 = BatchNormalization(axis=1)(dense1)
    # dnact1 = Activation('relu')(dnbn1)
    # dndrop1 = Dropout(0.2)(flt)

    output = Dense(17, activation='sigmoid')(bridge_pool41)

    _model = Model(inputs=input, outputs=output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
