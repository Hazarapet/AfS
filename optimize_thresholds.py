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

result = predict.result_single_tif(df_val['image_name'].values[:1000], 'resource/train-tif-v2/{}.tif')
thres = [0.32, 0.02, 0.24, 0.01, 0.94, 0.05, 0.03, 0.24, 0.24, 0.19, 0.81, 0.13, 0.52, 0.08, 0.01, 0.12, 0.31]

y = []
for tags in df_val['tags'].values[:1000]:
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1

    y.append(targets)

# use thresholds
p = []
for r in result:
    r = np.array(r).transpose()
    t = np.array(thres).transpose()
    p.append((r > t) * 1.0)

y = np.array(y).astype(np.float32)
p = np.array(p).astype(np.float32)
result = np.array(result).astype(np.float32)

print y.shape, p.shape, result.shape

# print result
print 'F2: ', common.f2_score(y, p).eval()

best_f2_threshold = common.optimise_f2_thresholds(y, result)

with open('best_f2_threshold.json', 'w') as outfile:
    json.dump(best_f2_threshold, outfile)

print '==== End ===='
