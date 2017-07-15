import sys
from keras.models import Model
from keras.layers import Input
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Activation


def model(weights_path=None):
    _input = Input((3, 256, 256))
    _m = ResNet50(weights=None, include_top=False, input_tensor=_input, input_shape=(3, 256, 256))
    _m.load_weights('models/nm/structures/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')

    # 175 layers
    for i, layer in enumerate(_m.layers):
        if i > 165:
            break

        layer.trainable = False

    x = _m.output
    x = Flatten(name='my_flatten_1')(x)
    x = Dropout(0.5, name='my_dp_flatten')(x)

    # x = Dense(1024, kernel_regularizer=l2(2e-5), name='my_dense_1')(x)
    # x = BatchNormalization(name='my_bn_1')(x)
    # x = Activation('relu', name='my_act_1')(x)
    # x = Dropout(0.4, name='my_dp_1')(x)

    # x = Dense(512, kernel_regularizer=l2(2e-5), name='my_dense_2')(x)
    # x = BatchNormalization(name='my_bn_2')(x)
    # x = Activation('relu', name='my_act_2')(x)
    # x = Dropout(0.4, name='my_dp_2')(x)

    x = Dense(17, name='my_output_dense')(x)
    x = Activation('sigmoid', name='my_output')(x)

    _model = Model(inputs=_m.input, outputs=x)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
