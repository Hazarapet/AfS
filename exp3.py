import sys
import numpy as np
import pandas as pd
import keras.backend as K
from utils import common as common_util

GROUP = ['artisinal_mine'
         'bare_ground',
         'blow_down',
         'conventional_mine',
         'cultivation',
         'haze',
         'selective_logging']

df_train = pd.read_csv('train_v2.csv')

label_map = {l: i for i, l in enumerate(GROUP)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in df_train.values[:5000]:
    targets = np.zeros(8)
    for t in tags.split(' '):
        if t in GROUP:
            targets[label_map[t]] = 1
        else:
            targets[7] = 1

        print targets

