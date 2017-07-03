import h5py
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Input, concatenate
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
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


def transition_block(input, nm_filter, block_index):
    prefix = 'transition_block_' + str(block_index)

    out = BatchNormalization(axis=1, name=prefix + '_bn1')(input)
    out = Activation('relu', name=prefix + '_relu1')(out)
    out = Conv2D(nm_filter, (1, 1), padding='same', use_bias=False, name=prefix + '_conv1')(out)

    out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=prefix + '_avg_pool1')(out)

    return out


def transition_bridge_block(input, nm_filter, block_index):
    prefix = 'transition_bridge_block_' + str(block_index)

    out = BatchNormalization(axis=1, name=prefix + '_bn1')(input)
    out = Activation('relu', name=prefix + '_relu1')(out)
    out = Conv2D(int(nm_filter * 0.5), (1, 1), padding='same', use_bias=False, name=prefix + '_conv1')(out)

    out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=prefix + '_avg_pool1')(out)

    return out


def dense_block(nb_layers, tmp_input, nm_filter, k, block_index):
    for i in range(nb_layers):
        conv = conv_block(input=tmp_input, nm_filter=k, dense_block_index=block_index, conv_block_index=i, dp=.33)
        tmp_input = concatenate([tmp_input, conv], axis=1, name='dense_block_' + str(block_index) + '_concat_' + str(i))

        nm_filter += k

    return tmp_input, nm_filter


def model(weights_path=None):
    k = 64
    nm_filter = 64
    compression = 0.5
    blocks = [3, 6]

    _input = Input((3, 224, 224))
    _resnet_50 = ResNet50(weights=None, include_top=False, input_tensor=_input, input_shape=(3, 224, 224))
    _resnet_50.load_weights('models/nm/structures/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')

    for layer in _resnet_50.layers:
        layer.trainable = False

    resnet_50 = _resnet_50.output
    resnet_50 = Flatten(name='my_flatten_1')(resnet_50)

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    tmp_input = resnet_50.get_layer('activation_22').output

    for i, block in enumerate(blocks):
        # prev_input = tmp_input
        tmp_input, nm_filter = dense_block(nb_layers=block, tmp_input=tmp_input, nm_filter=nm_filter, k=k, block_index=i)

        nm_filter = int(nm_filter * compression)

        if i < len(blocks) - 1:
            # TODO Every Dense block takes as input [transition_output, prev_dense_block_input]
            tmp_input = transition_block(input=tmp_input, nm_filter=nm_filter, block_index=i)
            # tmp_bridge_input = transition_bridge_block(input=prev_input, nm_filter=nm_filter, block_index=i, noise=0.01)

            # tmp_input = concatenate([tmp_input, tmp_bridge_input], axis=1)

    # -----------------------------------------------------
    # --------------------- Bridge ------------------------
    bridge = BatchNormalization(axis=1, name='bridge_bn1')(tmp_input)
    bridge = Activation('relu', name='bridge_relu1')(bridge)
    bridge = Flatten(name='bridge_flatten')(bridge)

    mini_dense = Dropout(0.5, name='bridge_dp1')(bridge)

    _output = concatenate([mini_dense, resnet_50])

    _model = Model(inputs=_input, outputs=_output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
