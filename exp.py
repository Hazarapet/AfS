import plots
import numpy as np

arr = np.random.random(18)
arr2 = np.random.random(18)
arr3 = np.random.random(18)
arr4 = np.random.random(18)
arr5 = np.random.random(18)
arr6 = np.random.random(18)
arr7 = np.random.random(18)
arr8 = np.random.random(18)

plots.plot_curve(values=[arr, arr2, arr3, arr4], labels=['Train Loss', 'Val Loss', 'Train F2', 'Val F2'], file_name='_plot.jpg')





