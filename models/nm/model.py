from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def conv_block(input, nm_filter, dp=0.1):
    out = BatchNormalization(axis=1)(input)
    out = Activation('relu')(out)
    out = Conv2D(4 * nm_filter, (1, 1), padding='same', use_bias=False)(out)

    out = BatchNormalization(axis=1)(out)
    out = Activation('relu')(out)
    out = Conv2D(nm_filter, (3, 3), padding='same', use_bias=False)(out)

    if dp:
        out = Dropout(dp)(out)

    return out

def transition_block(input, nm_filter):
    out = BatchNormalization(axis=1)(input)
    out = Activation('relu')(out)
    out = Conv2D(nm_filter, (1, 1), padding='same', use_bias=False)(out)

    out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(out)

    return out

def dense_block(nb_layers, tmp_input, nm_filter, k):
    for i in range(nb_layers):
        conv = conv_block(tmp_input, k)
        tmp_input = concatenate([tmp_input, conv], axis=1)

        nm_filter += k

    return tmp_input, nm_filter

def model(weights_path=None):
    k = 32
    nm_filter = 64
    compression = 0.5
    blocks = [6, 12, 24, 16]

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

    for ind in range(len(blocks) - 1):
        tmp_input, nm_filter = dense_block(nb_layers=blocks[ind], tmp_input=tmp_input, nm_filter=nm_filter, k=k)

        nm_filter = int(nm_filter * compression)

        tmp_input = transition_block(tmp_input, nm_filter)

    tmp_input, nm_filter = dense_block(nb_layers=blocks[-1], tmp_input=tmp_input, nm_filter=nm_filter, k=k)

    # -----------------------------------------------------
    # --------------------- Bridge 4 ----------------------
    bridge_bn41 = BatchNormalization(axis=1)(tmp_input)
    bridge_act41 = Activation('relu')(bridge_bn41)

    bridge_pool41 = GlobalAveragePooling2D()(bridge_act41)

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
