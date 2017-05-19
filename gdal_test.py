import cv2
import sys
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

tif_sample = 'resource/train-tif-sample/train_10064.tif'
jpg_sample = 'resource/train-jpg/train_10064.jpg'

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

    # normalized difference vegetation index (NDVI)
    ndvi = (nir - red)/(nir + red)

    # Normalized Difference Water Index (NDWI)
    ndwi = (green - nir)/(green + nir)

    # enhanced vegetation index (EVI)
    evi = 2.5 * (nir.astype(np.float32) - red.astype(np.float32))/(nir.astype(np.float32) + 6 * red.astype(np.float32) - 7.5 * blue.astype(np.float32) + 1)

    # soil-adjusted vegetation index (SAVI)
    savi = (1 + 0.5) * (nir - red)/(nir + red + 0.5)

    # Iron Oxide Ratio (IOR)
    # This band ratio highlights hydrothermally
    # altered rocks that have been subjected to oxidation of iron-bearing sulphides.
    ior = red / blue

    # Burn Area Index (BAI)
    bai = 1 / ((((0.1 - red) / 256) ** 2) * 256**2 + (((0.06 - nir) / 256) ** 2) * 256**2)

    # Leaf Area Index (LAI)
    lai = (3.618 * evi - 0.118)

    # Visible Atmospherically Resistant Index (VARI)
    vari = (green - red) / (green + red - blue)

    gs = gridspec.GridSpec(3, 3, right=0.9, left=0.1, hspace=0.2, wspace=0.1)

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

    ax = plt.subplot(gs[4], title='ior')
    ax.axis('off')
    ax.imshow(ior, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[5], title='bai')
    ax.axis('off')
    ax.imshow(bai, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[6], title='vari')
    ax.axis('off')
    ax.imshow(vari, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[7], title='lai')
    ax.axis('off')
    ax.imshow(lai, cmap='gray')
    ax.set_aspect('auto')

    plt.show()

