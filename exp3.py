import sys
import numpy as np
import pandas as pd

array = np.random.random((4, 8, 8))
array1 = array[:, :4, :4]
array2 = array[:, :4, 4:]
array3 = array[:, 4:, :4]
array4 = array[:, 4:, 4:]

print array.shape
print array1.shape

