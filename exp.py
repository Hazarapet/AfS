import numpy as np
from utils import common as common_util

array1 = np.arange(9)
array2 = 2 * np.arange(9)
array3 = 3 * np.arange(9)

[a1, a2] = common_util.parallel_shuffle(array1, array1)

print array1, a1
print array1, a2