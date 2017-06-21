import os
import cv2
import numpy as np
import math

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

    array = np.random.random((12, 3))

    list1 = []
    list2 = []

    list1.append([1, 2, 3, 6])
    list2.append([10, 20, 30, 60])

    print list1 + list2

