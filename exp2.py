import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


# img = cv2.imread('resource/train-jpg/{}.jpg'.format('train_30101'))
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# flip_h = np.flip(img, 1)
# flip_v = np.flip(img, 0)
# rt90 = np.rot90(img, 1, axes=(0, 1))
# rt90_flip_h = np.rot90(flip_h, 1, axes=(0, 1))
# rt90_flip_v = np.rot90(flip_v, 1, axes=(0, 1))
#
# plt.figure('flip_h')
# plt.imshow(flip_h)
# plt.figure('flip_v')
# plt.imshow(flip_v)
# plt.figure('rt90')
# plt.imshow(rt90)
# plt.figure('rt90_flip_h')
# plt.imshow(rt90_flip_h)
# plt.figure('rt90_flip_v')
# plt.imshow(rt90_flip_v)
# plt.figure('original')
# plt.imshow(img)
# plt.show()

with open('best_f2_threshold.json', 'r') as outfile:
    thresis = json.load(outfile)
    obj = {'score': 0.33, 'threshold': [1, 32, 3, 5], 'model': ''}
    thresis.append(obj)

with open('best_f2_threshold.json', 'w') as outfile:
    json.dump(thresis, outfile)

