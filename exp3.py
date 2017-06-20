import sys
import numpy as np
import pandas as pd
from utils import common as common_util
import h5py

df_train = pd.read_csv('train_v2.csv')

df_tr = pd.read_csv('train_split.csv')
df_val = pd.read_csv('val_split.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

train, val = df_tr.values, df_val.values

print labels
print label_map

for min_batch in common_util.iterate_minibatches(train[:40], batchsize=40):

    t_batch_inputs = []
    t_batch_labels = []

    # now we should load min_batch's images and collect them
    for f, tags in min_batch:
        targets = np.zeros(17)

        for t in tags.split(' '):
            targets[label_map[t]] = 1

        print tags, targets

