import cv2
import sys
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

tif_sample = 'resource/train-tif-sample/train_10033.tif'
jpg_sample = 'resource/train-jpg-sample/train_10033.jpg'
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

    gs = gridspec.GridSpec(2, 2, right=0.9, left=0.1, hspace=0.1, wspace=0.1)

    plt.figure('jpg')
    plt.imshow(jpg)
    plt.figure()

    ax = plt.subplot(gs[0], title='ndwi')
    ax.axis('off')
    ax.imshow(ndwi, cmap="gray")
    ax.set_aspect('auto')

    ax = plt.subplot(gs[1], title='ndvi')
    ax.axis('off')
    ax.imshow(ndvi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[2], title='evi')
    ax.axis('off')
    ax.imshow(evi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[3], title='savi')
    ax.axis('off')
    ax.imshow(savi, cmap='gray')
    ax.set_aspect('auto')

    plt.show()

