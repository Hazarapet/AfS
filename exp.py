import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import math, sys

a = range(3)

print sys.getsizeof(np.array([1], dtype='float16'))
print sys.getsizeof(np.array([1, 2], dtype="float16"))