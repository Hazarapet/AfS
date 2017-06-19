import os
import sys
import json
import predict
import numpy as np
import pandas as pd
import utils.common as common

# loading the data
df_val = pd.read_csv('val_split.csv')

df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

count = 0
print 'images loading...'

result = predict.result_single(df_val['image_name'].values, 'resource/train-jpg/{}.jpg')
thres = [0.05, 0.17, 0.05, 0.25, 0.32, 0.06, 0.1, 0.27, 0.28, 0.21, 0.09, 0.18, 0.16, 0.03, 0.2, 0.13, 0.04]  # Heng CherKeng's example

y = []
for tags in df_val['tags'].values:
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1

    y.append(targets)

p = []
for r in result:
    p.append(r > thres)

y = np.array(y).astype(np.float32)
p = np.array(p).astype(np.float32)
result = np.array(result).astype(np.float32)

# print result
print 'F2: ', common.f2_score(y, p).eval()

best_f2_threshold = common.optimise_f2_thresholds(y, result)

# print 'best threshold: ', best_f2_threshold
print '==== End ===='
