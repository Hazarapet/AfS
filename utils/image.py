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

    return tif


# normalized difference vegetation index (NDVI)
def ndvi(tif):
    red = tif[0]
    nir = tif[3]
    return (nir - red)/(nir + red)


# Normalized Difference Water Index (NDWI)
def ndwi(tif):
    nir = tif[3]
    green = tif[1]
    return (green - nir) / (green + nir)


# enhanced vegetation index (EVI)
def evi(tif):
    nir = tif[3]
    red = tif[0]
    blue = tif[2]

    G = 2.5
    C1 = 6.
    C2 = 7.5
    L = 1.

    return (G * (nir.astype(np.float32) - red.astype(np.float32))/(nir.astype(np.float32) + C1 * red.astype(np.float32) - C2 * blue.astype(np.float32) + L)).astype(np.float32)


# soil-adjusted vegetation index (SAVI)
def savi(tif):
    nir = tif[3]
    red = tif[0]
    L = 0.5
    return (1 + L) * (nir - red)/(nir + red + L)

# Iron Oxide Ratio (IOR)
# This band ratio highlights hydrothermally
# altered rocks that have been subjected to oxidation of iron-bearing sulphides.
def ior(tif):
    red = tif[0]
    blue = tif[1]
    return red / blue

# Burn Area Index (BAI)
def bai(tif):
    nir = tif[3]
    red = tif[0]
    return 1. / ((((0.1 - red) / 256.) ** 2) * 256.**2 + (((0.06 - nir) / 256.) ** 2) * 256.**2)

# Leaf Area Index (LAI)
def lai(tif):
    return 3.618 * evi(tif) - 0.118

# Visible Atmospherically Resistant Index (VARI)
def vari(tif):
    red = tif[0]
    blue = tif[1]
    green = tif[2]
    return (green - red) / (green + red - blue)

# Global Environmental Monitoring Index (GEMI)
def gemi(tif):
    nir = tif[3]
    red = tif[0]
    eta = 2 * ((((nir / 256.) ** 2) * (256. ** 2) - ((red / 256.) ** 2) * (256. ** 2)) + 1.5 * nir + 0.5 * red)/(nir + red + 0.5)
    return eta * (1 - 0.25 * eta) - (red - 0.125)/(1 - red)

# Green Difference Vegetation Index (GDVI)
def gndvi(tif):
    nir = tif[3]
    green = tif[1]
    return (nir - green) / (nir + green)

# Green Ratio Vegetation Index (GRVI)
def grvi(tif):
    nir = tif[3]
    green = tif[1]
    return nir / green

# Simple Ratio (SR)
def sr(tif):
    nir = tif[3]
    red = tif[0]
    return nir / red














