import sys
import numpy as np
import pandas as pd
import plots

array = np.array([2, 1, 3, 12, 5, 10, 7, 8, 9])
array2 = 10 * array - 5

plots.plot_curve([array, array2, array - 4], ['Loss', 'F2', 'DD'], 'test.jpg')

