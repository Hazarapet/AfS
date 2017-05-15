import sys
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt
from utils import image as image_util

tif_sample = 'resource/train-tif-sample/train_10051.tif'
jpg_sample = 'resource/train-jpg-sample/train_10051.jpg'

def read_tif(path):
    data = gdal.Open(path)
    rband = data.GetRasterBand(3)
    gband = data.GetRasterBand(2)
    bband = data.GetRasterBand(1)
    nirband = data.GetRasterBand(4)

    red = rband.ReadAsArray().astype(np.float32)
    green = gband.ReadAsArray().astype(np.float32)
    blue = bband.ReadAsArray().astype(np.float32)
    nir = nirband.ReadAsArray().astype(np.float32)

    return red, green, blue, nir

if __name__ == '__main__':
    red, green, blue, nir = read_tif(tif_sample)

    jpg = Image.open(jpg_sample)

    rgb = np.array([red, green, blue]).astype(np.int8)
    # mag, angle = image_util.img_sobel(green)

    ndvi = (nir - red)/(nir + red)

    ndwi = (green - nir)/(green + nir)

    evi = 2.5 * (nir - red)/(nir + 6 * red - 7.5 * blue + 1)

    savi = (1 + 0.5) * (nir - red)/(nir + red + 0.5)

    print 1

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

