import os
import sys
import time
import json
import numpy as np
import pandas as pd
import predict
import utils.common as common

st_time = time.time()

# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

count = 0
print 'images loading...'
X_test = os.listdir('resource/test-jpg')

result = predict.result_single_jpg(X_test, 'resource/test-jpg/{}', do_agg=False)
# Heng CherKeng's example
thres = [0.05, 0.17, 0.05, 0.25, 0.32, 0.06, 0.1, 0.27, 0.28, 0.21, 0.09, 0.18, 0.16, 0.03, 0.2, 0.13, 0.04]
# my_thres = [0.32, 0.02, 0.24, 0.01, 0.94, 0.05, 0.03, 0.24, 0.24, 0.19, 0.81, 0.13, 0.52, 0.08, 0.01, 0.12, 0.31]

with open('best_f2_threshold.json') as data_file:
    my_thres = json.load(data_file)
    my_thres = list(my_thres)

df_test = pd.DataFrame([[p.replace('.jpg', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

tags = []
result_b = []
for r in result:
    r = list(np.array(r).transpose() > my_thres)
    result_b.append(r)

result_b = np.array(result_b)

for sp in np.split(result_b, result_b.shape[0] / 3):
    r = common.ensemble(sp)
    t = [inv_label_map[i] for i, j in enumerate(r) if j]
    tags.append(' '.join(t))

df_test['tags'] = tags
df_test.to_csv('submission_0.csv', index=False)

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

