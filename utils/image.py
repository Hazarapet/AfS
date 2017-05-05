import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../resource/train-jpg/{}.jpg'.format('train_677'))
# img = img.transpose((2, 0, 1))
img = np.array(img, np.float32) / 255.

def img_sobel(img):
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return mag, angle

def img_cv2_std_dev(img):
    return cv2.meanStdDev(img)


