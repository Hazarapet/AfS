import os
import sys
import json
import time
import predict
import numpy as np
import pandas as pd
import utils.common as common

st_time = time.time()

# loading the data
df_val = pd.read_csv('val_split.csv')

df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# # 224x224, no norm, f2: 0.91917
weights_path_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.h5'
model_structure_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.json'

# # 224x224, no norm, f2: 0.93425 (whole db) ~0.921
weights_path_473_778 = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.h5'
model_structure_473_778 = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.json'

# 224x224, no norm, f2: 0.91892
weights_path_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.h5'
model_structure_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.json'

# 256x256, no, norm, f2: 0.92475
weights_path_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.h5'
model_structure_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.json'


result_459_14 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path_459_14,
                                   size=(224, 224),
                                   model_structure=model_structure_459_14)

result_473_778 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path_473_778,
                                   size=(224, 224),
                                   model_structure=model_structure_473_778)

result_600_446 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path_600_446,
                                   size=(224, 224),
                                   model_structure=model_structure_600_446)

result_386_612 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path_386_612,
                                   size=(256, 256),
                                   model_structure=model_structure_386_612)

y = []
for tags in df_val['tags'].values[:1000]:
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1

    y.append(targets)

# use thresholds
p = []
result = []
for r_600_446, r_473_778, r_459_14, r_386_612 in zip(result_600_446, result_473_778, result_459_14, result_386_612):
    r_600_446 = 1. * np.array(r_600_446)
    r_473_778 = 3. * np.array(r_473_778)
    r_459_14 = 2. * np.array(r_459_14)
    r_386_612 = 5. * np.array(r_386_612)

    r = np.sum([r_600_446, r_473_778, r_459_14, r_386_612], axis=0) / 11.
    result.append(r)

y = np.array(y).astype(np.float32)
result = np.array(result).astype(np.float32)

print y.shape, result.shape

best_f2_threshold, best_score = common.optimise_f2_thresholds(y, result)

model_name = 'ensemble-600_446-473_778-459_14-386_612-weighting-1_3_2_5'

with open('best_f2_threshold.json', 'r') as outfile:
    thresis = json.load(outfile)
    obj = {'score': best_score, 'threshold': best_f2_threshold, 'model': model_name}
    thresis.append(obj)

with open('best_f2_threshold.json', 'w') as outfile:
    json.dump(thresis, outfile)

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '==== End ===='
