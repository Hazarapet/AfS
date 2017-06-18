import sys
import numpy as np
import pandas as pd
import h5py

f = h5py.File("models/nm/structures/densenet121_weights_th.h5", "r")

for k, v in zip(f.keys(), f.values()):
    print v.items()

