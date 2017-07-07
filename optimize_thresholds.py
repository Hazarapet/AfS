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

weights_path = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.h5'
model_structure = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.json'

result = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path,
                                   model_structure=model_structure)

thres1 = [0.14, 0.15, 0.14, 0.27, 0.1, 0.35, 0.17, 0.32, 0.24, 0.12, 0.08, 0.1, 0.12, 0.13, 0.18, 0.16, 0.31]

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
    t = np.array(thres1).transpose()
    p.append((r > t) * 1.0)

y = np.array(y).astype(np.float32)
p = np.array(p).astype(np.float32)
result = np.array(result).astype(np.float32)

print y.shape, p.shape, result.shape

# print result
print 'F2: ', common.f2_score(y, p).eval()

best_f2_threshold, best_score = common.optimise_f2_thresholds(y, result)

with open('best_f2_threshold.json', 'r') as outfile:
    thresis = json.load(outfile)
    obj = {'score': best_score, 'threshold': best_f2_threshold, 'model': weights_path}
    thresis.append(obj)

with open('best_f2_threshold.json', 'w') as outfile:
    json.dump(thresis, outfile)

print '==== End ===='
