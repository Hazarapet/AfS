import cv2
import sys
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt
from utils import image as image_util
from sklearn.preprocessing import MinMaxScaler

tif_sample = 'resource/train-tif-sample/train_10020.tif'
jpg_sample = 'resource/train-jpg-sample/train_10020.jpg'

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

    print np.max(red), np.min(red)

    # img = Image.fromarray(np.uint8(rgb), "RGB")

    plt.figure('jpg')
    plt.imshow(jpg)
    plt.figure('ndwi')
    plt.imshow(ndwi, cmap="gray")
    plt.figure('ndvi')
    plt.imshow(ndvi, cmap='gray')
    plt.figure('evi')
    plt.imshow(evi, cmap='gray')
    plt.figure('savi')
    plt.imshow(savi, cmap='gray')
    plt.show()

