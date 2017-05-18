import sys
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

A = np.arange(27).reshape((3, 3, 3))
B = np.arange(9).reshape((3, 3))

A1 = np.rot90(A, axes=(1, 2))
A2 = np.flip(A, 2)

print A, 'end'
print A2, 'end'
