import os
import sys
import json
import time
import predict
import numpy as np
import pandas as pd
import utils.common as common

st_time = time.time()

#  loading the data
df_val = pd.read_csv('val_split.csv')

df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

thres_345_134 = [0.27, 0.21, 0.56, 0.08, 0.43, 0.62, 0.3, 0.59, 0.38, 0.08, 0.19, 0.19, 0.44, 0.33, 0.16, 0.19, 0.51]
thres_459_14 = [0.19, 0.09, 0.67, 0.29, 0.55, 0.91, 0.16, 0.4, 0.32, 0.1, 0.21, 0.14, 0.28, 0.17, 0.13, 0.11, 0.18]
thres_473_778 = [0.15, 0.17, 0.44, 0.14, 0.41, 0.63, 0.2, 0.18, 0.28, 0.27, 0.34, 0.2, 0.2, 0.09, 0.23, 0.2, 0.32]
thres_600_446 = [0.14, 0.15, 0.14, 0.27, 0.1, 0.35, 0.17, 0.32, 0.24, 0.12, 0.08, 0.1, 0.12, 0.13, 0.18, 0.16, 0.31]
thres_386_612 = [0.1, 0.13, 0.43, 0.21, 0.25, 0.47, 0.14, 0.63, 0.29, 0.19, 0.15, 0.17, 0.24, 0.1, 0.29, 0.13, 0.39]

# # # 224x224, no norm, f2: 0.91917
# weights_path_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.h5'
# model_structure_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.json'
#
# # # 224x224, no norm, f2: 0.93425 (whole db) ~0.921
# weights_path_473_778 = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.h5'
# model_structure_473_778 = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.json'
#
# # 224x224, no norm, f2: 0.91892
# weights_path_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.h5'
# model_structure_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.json'

# 256x256, no, norm, f2: 0.92475
weights_path_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.h5'
model_structure_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.json'


# result_459_14 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
#                                    path='resource/train-jpg/{}.jpg',
#                                    weights_path=weights_path_459_14,
#                                    size=(224, 224),
#                                    model_structure=model_structure_459_14)
#
# result_473_778 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
#                                    path='resource/train-jpg/{}.jpg',
#                                    weights_path=weights_path_473_778,
#                                    size=(224, 224),
#                                    model_structure=model_structure_473_778)
#
# result_600_446 = predict.result_single_jpg(X=df_val['image_name'].values[:1000],
#                                    path='resource/train-jpg/{}.jpg',
#                                    weights_path=weights_path_600_446,
#                                    size=(224, 224),
#                                    model_structure=model_structure_600_446)

result_386_612 = predict.result_single_jpg(X=df_val['image_name'].values[:2000],
                                   path='resource/train-jpg/{}.jpg',
                                   weights_path=weights_path_386_612,
                                   size=(256, 256),
                                   verbose=False,
                                   model_structure=model_structure_386_612)

y = []
for tags in df_val['tags'].values[:2000]:
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1

    y.append(targets)

# use thresholds
result = result_386_612
p = []
for r in result:
    r = np.array(r).transpose()
    t = np.array(thres_386_612).transpose()
    p.append((r > t) * 1.0)

y = np.array(y).astype(np.float32)
p = np.array(p).astype(np.float32)
result = np.array(result).astype(np.float32)

print y.shape, p.shape, result.shape

errors = []
for pr, r in zip(p, y):
    err = pr != r
    errors.append(err * 1.)

errors = np.array(errors)
errors_sum = np.sum(errors, axis=0)
errors_sum_tags = {inv_label_map[i]: j for i, j in enumerate(errors_sum)}

print np.sum(errors_sum)
print errors_sum_tags

with open('errors.json', 'r') as outfile:
    thresis = json.load(outfile)
    obj = {'errors': np.sum(errors_sum), 'tags': errors_sum_tags, 'model': weights_path_386_612}
    thresis.append(obj)

with open('errors.json', 'w') as outfile:
    json.dump(thresis, outfile)


print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '==== End ===='
