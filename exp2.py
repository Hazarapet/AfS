import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('resource/train-jpg/{}.jpg'.format('train_10129'))

img = img.transpose((2, 0, 1))

print 'first: ', img.shape

rt90 = np.rot90(img, 1, axes=(1, 2))
flip_v = np.flip(img, 2)
flip_h = np.flip(img, 1)

img = img.transpose((1, 2, 0))
rt90 = rt90.transpose((1, 2, 0))
flip_v = flip_v.transpose((1, 2, 0))
flip_h = flip_h.transpose((1, 2, 0))

print 'modified: ', rt90.shape, flip_v.shape, flip_h.shape
print 'second: ', img.shape

plt.figure('rot')
plt.imshow(rt90)
plt.figure('flip_h')
plt.imshow(flip_h)
plt.figure('flip_v')
plt.imshow(flip_v)
plt.figure('org')
plt.imshow(img)
plt.show()
