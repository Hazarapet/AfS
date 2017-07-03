import h5py
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Input, concatenate
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

K.set_image_data_format('channels_first')

def model(weights_path=None):
    _m = ResNet50(weights='imagenet', include_top=False)

    for layer in _m.layers:
        layer.trainable = False

    x = _m.output
    x = Dense(512, name='my_dense_1')(x)
    x = BatchNormalization(axis=1, name='my_bn_1')(x)
    x = Activation('relu', name='my_act_1')(x)

    x = Dense(17, name='my_dense_1')(x)
    x = Activation('sigmout', name='my_output')(x)

    _model = Model(inputs=_m.input, outputs=x)

    return [_model, 'models/nm/structures/']