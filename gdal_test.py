import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt
from utils import image as image_util

tif_sample = 'resource/train-tif-sample/train_10008.tif'
jpg_sample = 'resource/train-jpg-sample/train_10008.jpg'

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

    rgb = np.array([red, green, blue])
    rgb = rgb.transpose((1, 2, 0))
    mag, angle = image_util.img_sobel(green)

    ndvi = (nir - red)/(nir + red)

    print np.max(red), np.max(ndvi)

    plt.figure('jpg')
    plt.imshow(jpg)
    # plt.figure('blooming')
    # plt.imshow(mag, cmap="gray")
    plt.figure('ndvi')
    plt.imshow(ndvi, cmap='gray')
    plt.show()

