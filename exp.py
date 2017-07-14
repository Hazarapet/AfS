import plots
import numpy as np

arr1 = np.random.random((3, 128, 128))
arr2 = np.random.random((128, 128))

arr1 = np.append(arr1, [arr2], axis=0)

print arr1.shape





