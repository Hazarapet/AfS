import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('resource/train-jpg/{}.jpg'.format('train_696'))
img = np.array(img, np.float32) / 255.

# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(mag)

plt.show()
