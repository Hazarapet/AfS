import os
import cv2
import numpy as np


def ensemble(array):
    new_array = []
    for cl in range(array.shape[1]):
        cn = list(array[:, cl]).count(1)
        all_cn = array.shape[0]
        if cn > all_cn / 2:
            new_array.append(1)
        else:
            new_array.append(0)
    return new_array

if __name__ == '__main__':

    array = np.ones((4, 3)) * (np.random.random((4, 3)) > 0.4) * 1
    array = array.astype(np.uint8)
    ens = ensemble(array)

    print array
    print ens

