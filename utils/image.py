import cv2
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt

def sobel(img):
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return mag, angle

def cv2_std_dev(img):
    return cv2.meanStdDev(img)

def process_tif(path):
    data = gdal.Open(path)
    rband = data.GetRasterBand(3)
    gband = data.GetRasterBand(2)
    bband = data.GetRasterBand(1)
    nirband = data.GetRasterBand(4)

    red = rband.ReadAsArray().astype(np.float32)
    green = gband.ReadAsArray().astype(np.float32)
    blue = bband.ReadAsArray().astype(np.float32)
    nir = nirband.ReadAsArray().astype(np.float32)

    tif = [red, green, blue, nir]

    return tif, ndwi(tif), ndvi(tif), evi(tif), savi(tif)

def ndvi(tif):
    red = tif[0]
    nir = tif[3]
    return (nir - red)/(nir + red)

def ndwi(tif):
    nir = tif[3]
    green = tif[1]
    return (green - nir) / (green + nir)

def evi(tif):
    nir = tif[3]
    red = tif[0]
    blue = tif[2]

    G = 2.5
    C1 = 6
    C2 = 7.5
    L = 1

    return G * (nir - red)/(nir + C1 * red - C2 * blue + L)

def savi(tif):
    nir = tif[3]
    red = tif[0]
    L = 0.5
    return (1 + L) * (nir - red)/(nir + red + L)

