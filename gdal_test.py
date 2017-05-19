import cv2
import sys
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

tif_sample = 'resource/train-tif-v2/train_9991.tif'
jpg_sample = 'resource/train-jpg/train_9991.jpg'

def read_tif(path):
    data = gdal.Open(path)
    rband = data.GetRasterBand(3)
    gband = data.GetRasterBand(2)
    bband = data.GetRasterBand(1)
    nirband = data.GetRasterBand(4)

    red = rband.ReadAsArray().astype(np.float16)
    green = gband.ReadAsArray().astype(np.float16)
    blue = bband.ReadAsArray().astype(np.float16)
    nir = nirband.ReadAsArray().astype(np.float16)

    return red, green, blue, nir

if __name__ == '__main__':
    red, green, blue, nir = read_tif(tif_sample)

    jpg = Image.open(jpg_sample)

    rgb = np.array([red, green, blue])
    # rgb = rgb.transpose((1, 2, 0))
    # rescaleIMG = np.reshape(rgb, (-1, 1))
    # scaler = MinMaxScaler(feature_range=(0, 255))
    # rescaleIMG = scaler.fit_transform(rescaleIMG)  # .astype(np.float32)
    # img2_scaled = (np.reshape(rescaleIMG, rgb.shape)).astype(np.uint8)
    # img2_scaled = img2_scaled.transpose((1, 2, 0))

    ndvi = (nir - red)/(nir + red)

    ndwi = (green - nir)/(green + nir)

    evi = 2.5 * (nir - red)/(nir + 6 * red - 7.5 * blue + 1)

    savi = (1 + 0.5) * (nir - red)/(nir + red + 0.5)

    plt.figure('jpg')
    plt.imshow(jpg)
    plt.figure()

    plt.subplot(221, title='ndwi')
    plt.imshow(ndwi, cmap="gray")
    plt.axis('off')
    plt.tight_layout(w_pad=0.2, h_pad=0.2)

    plt.subplot(222, title='ndvi')
    plt.axis('off')
    plt.imshow(ndvi, cmap='gray')

    plt.tight_layout(w_pad=0.2, h_pad=0.2)

    plt.subplot(223, title='evi')
    plt.axis('off')
    plt.imshow(evi, cmap='gray')

    plt.subplot(224, title='savi')
    plt.axis('off')
    plt.imshow(savi, cmap='gray')

    plt.show()

