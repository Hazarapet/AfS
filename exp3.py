import sys
import h5py
import numpy as np
import pandas as pd


def load_weights(_model, weights_path):
    f = h5py.File(weights_path, "r")

    weights = {k: v for k, v in zip(f.keys(), f.values())}
    layer_name = 'conv1'
    # for l in weights:
    #     print l, weights[l].items()
    # for layer in _model.layers:
    for w in weights:
        # w = weights[layer_name]
        ar = []
        for v in weights[w].values():
            print '------', v.shape, np.array(v[()]).shape
            # for vv in v.values():
            # ar.append(np.array(v[()]))

        # ar = np.array(ar).astype(np.float32)
        # layer.set_weights(ar)

        # print ar.shape
        # print ar
        print 'layer "' + w + '" weights are loaded'

    return _model


if __name__ == '__main__':

    # load_weights(None, weights_path='models/nm/structures/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')
    print 'conv1----------------------------------------------------------------------------------------------------'
    print 'conv1----------------------------------------------------------------------------------------------------'
    print 'conv1----------------------------------------------------------------------------------------------------'
    load_weights(None, weights_path='models/nm/structures/densenet121_weights_th.h5')





