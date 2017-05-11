import gdal
import numpy as np
from gdalconst import *

tif_sample = 'resource/train-tif-sample/train_0.tif'

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
    red = rband.ReadAsArray()
    print red