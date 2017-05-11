import gdal
import numpy as np
from gdalconst import *
import matplotlib.pyplot as plt
from utils import image as image_util

tif_sample = 'resource/train-tif-sample/train_10087.tif'

def read_gdalImg(path):
    data = gdal.Open(path)
    rband = data.GetRasterBand(1)
    gband = data.GetRasterBand(2)
    bband = data.GetRasterBand(3)
    nirband = data.GetRasterBand(4)
    red = rband.ReadAsArray().astype(np.uint8)
    green = gband.ReadAsArray().astype(np.uint8)
    blue = bband.ReadAsArray().astype(np.uint8)
    nir = nirband.ReadAsArray().astype(np.uint8)
    return red, green, blue, nir

if __name__ == '__main__':
    data = gdal.Open(tif_sample)
    rband = data.GetRasterBand(1)
    gband = data.GetRasterBand(2)
    bband = data.GetRasterBand(3)
    nirband = data.GetRasterBand(4)
    
    red = rband.ReadAsArray()
    green = gband.ReadAsArray()
    blue = bband.ReadAsArray()
    nir = nirband.ReadAsArray()

    rgb = np.array([red, green, blue])
    rgb = rgb.transpose((1, 2, 0))
    mag, angle = image_util.img_sobel(green)
    ndvi = (nir - red)/(nir + red)

    print np.max(green)

    plt.figure('green')
    plt.imshow(green, cmap='gray')
    plt.figure('blooming')
    plt.imshow(mag > 500, cmap="gray")
    plt.show()

