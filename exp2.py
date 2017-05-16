import json
import cv2
from PIL import Image
import numpy as np

A = np.arange(27).reshape((3, 3, 3))
B = np.fliplr(A)
print A
print B

