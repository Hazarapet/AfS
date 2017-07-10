import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate
from keras.applications.vgg16 import VGG16
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


def transition_connect_block(input, nm_filter, block_index, str_name=''):
    prefix = 'transition_connect_block_' + str(block_index) + '_' + str_name

    out = BatchNormalization(axis=1, name=prefix + '_bn1')(input)
    out = Activation('relu', name=prefix + '_relu1')(out)
    out = Conv2D(int(nm_filter * 0.4), (1, 1), padding='same', use_bias=False, name=prefix + '_conv1')(out)

    return out


def dense_block(nb_layers, tmp_input, nm_filter, k, block_index):
    for i in range(nb_layers):
        conv = conv_block(input=tmp_input, nm_filter=k, dense_block_index=block_index, conv_block_index=i, dp=.3)
        tmp_input = concatenate([tmp_input, conv], axis=1, name='dense_block_' + str(block_index) + '_concat_' + str(i))

        nm_filter += k

    return tmp_input, nm_filter


def model(weights_path=None):
    k = 12
    nm_filter = 64
    compression = 0.5
    blocks = [6, 12, 24, 16]

    # TODO 64, 32, 16, 8
    _resnet50_outputs = ['activation_10', 'activation_22', 'activation_40', 'activation_49']
    _vgg16_outputs = ['block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

    # _input_128 = Input((3, 128, 128))
    _input_257 = Input((3, 257, 257))
    _input_256 = Input((3, 256, 256))

    # -----------------------------------------------------
    # ----------------- ResNet50 Freeze -------------------
    _resnet50_freeze = ResNet50(weights=None, include_top=False, input_tensor=_input_257, input_shape=(3, 257, 257))
    _resnet50_freeze.load_weights('models/nm/structures/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')

    for layer in _resnet50_freeze.layers:
        layer.trainable = False

    # -----------------------------------------------------
    # ------------------ Vgg16 Freeze ---------------------
    # _vgg16_freeze = VGG16(weights=None, include_top=False, input_tensor=_input_128, input_shape=(3, 128, 128))
    # _vgg16_freeze.load_weights('models/main/structures/vgg16_weights_th_dim_ordering_th_kernels_notop.h5')
    #
    # for layer in _vgg16_freeze.layers:
    #     layer.trainable = False

    # ------------------------------------------------------
    # ------------------------------------------------------
    # ------------------------------------------------------
    start_conv = ZeroPadding2D((3, 3), name='gateway_padding3x3')(_input_256)
    start_conv = Conv2D(nm_filter, (7, 7), strides=(2, 2), name='gateway_conv', use_bias=False)(start_conv)
    start_conv = BatchNormalization(axis=1, name='gateway_bn')(start_conv)
    start_conv = Activation('relu', name='gateway_act')(start_conv)

    start_conv = ZeroPadding2D((1, 1), name='gateway_padding1x1')(start_conv)
    start_conv = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='gateway_max_pool')(start_conv)

    # ------------------------------------------------------
    # ---------------------- 64x64 -------------------------
    # ------------------ Conv Block 1 ----------------------
    tmp_input = start_conv

    for i, block in enumerate(blocks):
        if i < len(_vgg16_outputs) and i < len(_resnet50_outputs):
            # _vgg16_connect = transition_connect_block(input=_vgg16_freeze.get_layer(_vgg16_outputs[i]).output, nm_filter=nm_filter, block_index=i, str_name='vgg16')
            _resnet50_connect = transition_connect_block(input=_resnet50_freeze.get_layer(_resnet50_outputs[i]).output, nm_filter=nm_filter, block_index=i, str_name='resnet50')

            tmp_input = concatenate([tmp_input, _resnet50_connect], axis=1)

        tmp_input, nm_filter = dense_block(nb_layers=block, tmp_input=tmp_input, nm_filter=nm_filter, k=k, block_index=i)

        nm_filter = int(nm_filter * compression)

        if i < len(blocks) - 1:
            tmp_input = transition_block(input=tmp_input, nm_filter=nm_filter, block_index=i)

    # -----------------------------------------------------
    # --------------------- Bridge ------------------------

    bridge = BatchNormalization(axis=1, name='bridge_bn1')(tmp_input)
    bridge = Activation('relu', name='bridge_relu1')(bridge)
    bridge = GlobalAveragePooling2D(name='bridge_glb_avg_pool')(bridge)
    # bridge = Flatten(name='bridge_flatten')(bridge)

    _resnet50_output = Flatten(name='my_resnet_flatten')(_resnet50_freeze.output)
    _resnet50_output = BatchNormalization(name='my_resnet_bn1')(_resnet50_output)
    _resnet50_output = Activation('relu', name='my_resnet_relu1')(_resnet50_output)

    # _vgg16_output = Flatten(name='my_vgg_flatten')(_vgg16_freeze.output)
    # _vgg16_output = BatchNormalization(name='my_vgg_bn1')(_vgg16_output)
    # _vgg16_output = Activation('relu', name='my_vgg_relu1')(_vgg16_output)

    _concat = concatenate([bridge, _resnet50_output])

    _output = Dropout(0.5, name='my_dp_1')(_concat)

    _output = Dense(17, name='my_output_dense_2')(_output)
    _output = Activation('sigmoid', name='output')(_output)

    _model = Model(inputs=[_input_256, _input_257], outputs=_output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
