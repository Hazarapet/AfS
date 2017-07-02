import h5py
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def conv_block(input, nm_filter, dense_block_index, conv_block_index, dp=0.3):
    prefix = 'dense_block_' + str(dense_block_index) + '_conv_block_' + str(conv_block_index)

    out = BatchNormalization(axis=1, name=prefix + '_bn1')(input)
    out = Activation('relu', name=prefix + '_relu1')(out)
    out = Conv2D(4 * nm_filter, (1, 1), padding='same', use_bias=False, name=prefix + '_conv1')(out)

    if dp:
        out = Dropout(dp, name=prefix + '_dp1')(out)

    out = BatchNormalization(axis=1, name=prefix + '_bn2')(out)
    out = Activation('relu', name=prefix + '_relu2')(out)
    out = Conv2D(nm_filter, (3, 3), padding='same', use_bias=False, name=prefix + '_conv2')(out)

    if dp:
        out = Dropout(dp, name=prefix + '_dp2')(out)

    return out


def transition_block(input, nm_filter, block_index, noise=.0):
    prefix = 'transition_block_' + str(block_index)

    out = BatchNormalization(axis=1, name=prefix + '_bn1')(input)
    out = Activation('relu', name=prefix + '_relu1')(out)
    out = Conv2D(nm_filter, (1, 1), padding='same', use_bias=False, name=prefix + '_conv1')(out)

    out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=prefix + '_avg_pool1')(out)

    if noise:
        out = GaussianNoise(noise, name=prefix + '_gn_noise1')(out)

    return out


def transition_bridge_block(input, nm_filter, block_index, noise=.0):
    prefix = 'transition_bridge_block_' + str(block_index)

    out = BatchNormalization(axis=1, name=prefix + '_bn1')(input)
    out = Activation('relu', name=prefix + '_relu1')(out)
    out = Conv2D(int(nm_filter * 0.5), (1, 1), padding='same', use_bias=False, name=prefix + '_conv1')(out)

    out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=prefix + '_avg_pool1')(out)

    if noise:
        out = GaussianNoise(noise, name=prefix + '_gn_noise1')(out)

    return out


def dense_block(nb_layers, tmp_input, nm_filter, k, block_index):
    for i in range(nb_layers):
        conv = conv_block(input=tmp_input, nm_filter=k, dense_block_index=block_index, conv_block_index=i, dp=.33)
        tmp_input = concatenate([tmp_input, conv], axis=1, name='dense_block_' + str(block_index) + '_concat_' + str(i))

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
    start_conv = Conv2D(nm_filter, (7, 7), strides=(2, 2), name='gateway_conv', use_bias=False)(start_conv)
    start_conv = BatchNormalization(axis=1, name='gateway_bn')(start_conv)
    start_conv = Activation('relu', name='gateway_act')(start_conv)

    start_conv = ZeroPadding2D((1, 1), name='gateway_padding1x1')(start_conv)
    start_conv = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='gateway_max_pool')(start_conv)

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    tmp_input = start_conv

    for i, block in enumerate(blocks):
        # prev_input = tmp_input
        tmp_input, nm_filter = dense_block(nb_layers=block, tmp_input=tmp_input, nm_filter=nm_filter, k=k, block_index=i)

        nm_filter = int(nm_filter * compression)

        if i < len(blocks) - 1:
            # TODO Every Dense block takes as input [transition_output, prev_dense_block_input]
            tmp_input = transition_block(input=tmp_input, nm_filter=nm_filter, block_index=i)
            # tmp_bridge_input = transition_bridge_block(prev_input, nm_filter)
            #
            # tmp_input = concatenate([tmp_input, tmp_bridge_input], axis=1)

    # -----------------------------------------------------
    # --------------------- Bridge ------------------------
    bridge = BatchNormalization(axis=1, name='bridge_bn1')(tmp_input)
    bridge = Activation('relu', name='bridge_relu1')(bridge)

    bridge = GlobalAveragePooling2D(name='bridge_global_avg1')(bridge)
    # bridge = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='bridge_max_pool1')(bridge)

    # bridge = Flatten(name='bridge_flatten')(bridge)

    output = Dense(17, activation='sigmoid', name='bridge_sigmoid1')(bridge)

    _model = Model(inputs=input, outputs=output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']


def load_weights(_model, weights_path):
    f = h5py.File(weights_path, "r")

    weights = {k: v for k, v in zip(f.keys(), f.values())}

    for layer in _model.layers:
        if layer.name in weights:
            w = weights[layer.name]
            ar = []
            for v in w.values():
                for vv in v.values():
                    ar.append(vv[()])

            ar = np.array(ar).astype(np.float32)
            layer.set_weights(ar)

            print 'layer "' + layer.name + '" weights are loaded'

    return _model
