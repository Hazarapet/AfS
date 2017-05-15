import json
import cv2
from PIL import Image
import numpy as np

A = np.arange(27).reshape((3, 3, 3))

print A
print np.flip(A, 2)

