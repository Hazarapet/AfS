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
X_test = os.listdir('resource/test-jpg')

result = predict.result_single_jpg(X_test, 'resource/test-jpg/{}')
thres = [0.05, 0.17, 0.05, 0.25, 0.32, 0.06, 0.1, 0.27, 0.28, 0.21, 0.09, 0.18, 0.16, 0.03, 0.2, 0.13, 0.04]  # Heng CherKeng's example
my_thres = [0.32, 0.13, 0.96, 0.05, 0.6, 0.96, 0.08, 0.24, 0.12, 0.11, 0.31, 0.08, 0.6, 0.08, 0.05, 0.11, 0.98]  # Heng CherKeng's example

df_test = pd.DataFrame([[p.replace('.tif', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

tags = []
for r in result:
    r = list(r > my_thres)
    t = [inv_label_map[i] for i, j in enumerate(r) if j]
    tags.append(' '.join(t))

df_test['tags'] = tags
df_test.to_csv('submission_0.csv', index=False)

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

