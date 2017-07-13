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

weights_path = 'models/nm/structures/tr_l:0.1174-tr_f2:0.9148-val_l:0.1245-val_f2:0.9098-time:13-07-2017-08:27:59-dur:630.577.h5'
model_structure = 'models/nm/structures/tr_l:0.1174-tr_f2:0.9148-val_l:0.1245-val_f2:0.9098-time:13-07-2017-08:27:59-dur:630.577.json'

result = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path,
                                   model_structure=model_structure)

thres = [0.1, 0.13, 0.43, 0.21, 0.25, 0.47, 0.14, 0.63, 0.29, 0.19, 0.15, 0.17, 0.24, 0.1, 0.29, 0.13, 0.39]  # test

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

best_f2_threshold, best_score = common.optimise_f2_thresholds(y, result)

# with open('best_f2_threshold.json', 'r') as outfile:
#     thresis = json.load(outfile)
#     obj = {'score': best_score, 'threshold': best_f2_threshold, 'model': weights_path}
#     thresis.append(obj)
#
# with open('best_f2_threshold.json', 'w') as outfile:
#     json.dump(thresis, outfile)

print '==== End ===='
