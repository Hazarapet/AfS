import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('resource/train-jpg/{}.jpg'.format('train_13101'))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
rn1 = np.random.randint(0, 32)
rn2 = np.random.randint(img.shape[0] - 32, img.shape[0])

print 'image shape: ', img.shape, rn1, rn2

crop_img = cv2.resize(img[rn1:rn2, rn1:rn2], (256, 256))  # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
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
plt.figure('cropped')
plt.imshow(crop_img)
plt.figure('original')
plt.imshow(img)
plt.show()

