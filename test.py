import os
import sys
import time
import numpy as np
import pandas as pd
import predict

st_time = time.time()

# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

count = 0
print 'images loading...'
X_test = os.listdir('resource/test-tif-v2')

result = predict.result(X_test, 'resource/test-tif-v2/{}')
thres = [0.085, 0.2375, 0.19, 0.5, 0.16, 0.0875, 0.5, 0.1925, 0.265, 0.1625, 0.1375, 0.2175, 0.2225, 0.0475, 0.5, 0.5, 0.14]  # Heng CherKeng's example

df_test = pd.DataFrame([[p.replace('.tif', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

tags = []
for r in result:
    r = list(r > .2)
    t = [inv_label_map[i] for i, j in enumerate(r) if j]
    tags.append(' '.join(t))

df_test['tags'] = tags
df_test.to_csv('submission_0.csv', index=False)

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

