from keras.models import Model
from keras.layers import Input
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization


def model(weights_path=None):
    _input = Input((3, 224, 224))
    _m = VGG16(weights=None, include_top=False, input_tensor=_input, input_shape=(3, 224, 224))
    _m.load_weights('models/main/structures/vgg16_weights_th_dim_ordering_th_kernels_notop.h5')

    # 19
    for i, layer in enumerate(_m.layers):
        if i > 10:
            break

        layer.trainable = False

    x = _m.output
    x = Flatten(name='my_flatten_1')(x)

    x = Dense(1024, kernel_regularizer=l2(2e-5), name='my_dense_1')(x)
    x = BatchNormalization(name='my_bn_1')(x)
    x = Activation('relu', name='my_act_1')(x)
    x = Dropout(0.5, name='my_dp_1')(x)

    x = Dense(17, name='my_output_dense')(x)
    x = Activation('sigmoid', name='my_output')(x)

    _model = Model(inputs=_m.input, outputs=x)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/main/structures/']

