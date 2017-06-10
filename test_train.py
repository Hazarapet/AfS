import os
import sys
import json
import predict
import numpy as np
import pandas as pd
import optimize_f2
import utils.common as common

# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

count = 0
print 'images loading...'
X_train = os.listdir('resource/train-tif-v2')

result = predict.result(df_train['image_name'].values[:1000], 'resource/train-tif-v2/{}.tif')
thres = [0.085, 0, 0.19, 0.5, 0.16, 0.0875, 0.5, 0.1925, 0.265, 0.1625, 0.1375, 0.2175, 0.2225, 0.0475, 0.5, 0.5, 0.14]  # Heng CherKeng's example

y = []
for tags in df_train['tags'].values[:1000]:
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

with open('results.json', 'w') as outfile:
    rs = {'p': result.tolist(), 'y': y.tolist()}
    json.dump(rs, outfile)

# best_f2_threshold = optimize_f2.optimise_f2_thresholds(y, result)

# print 'best threshold: ', best_f2_threshold
print '==== End ===='
