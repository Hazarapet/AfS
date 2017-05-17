import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

y = [ [0, 1, 2, 3, 4], [5, 6, 7, 8, 19], [9, 8, 7, 6, 5] ]
labels = ['foo', 'bar', 'baz']

for y_arr, label in zip(y, labels):
    plt.plot(y_arr, label=label)

plt.legend(bbox_to_anchor=(1, 1), loc=5)
plt.show()