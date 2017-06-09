import sys
import numpy as np
import pandas as pd

GROUP = ['artisinal_mine',
         'bare_ground',
         'blooming',
         'blow_down',
         'conventional_mine',
         'cultivation',
         'haze',
         'selective_logging']

labels = GROUP

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print label_map
print inv_label_map


# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print label_map
print inv_label_map

