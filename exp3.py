import sys
import numpy as np
import pandas as pd
import plots

array = np.ones(12).reshape((4, 3))

print np.sum(array, axis=0) / 4

