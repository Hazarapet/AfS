import cv2
import sys
import gdal
import numpy as np
from PIL import Image
from gdalconst import *
import utils.image as image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

tif_sample = 'resource/train-tif-sample/train_10081.tif'
jpg_sample = 'resource/train-jpg/train_10081.jpg'

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
    rgbn = np.array([red, green, blue, nir])
    # rgb = rgb.transpose((1, 2, 0))
    # rescaleIMG = np.reshape(rgb, (-1, 1))
    # scaler = MinMaxScaler(feature_range=(0, 255))
    # rescaleIMG = scaler.fit_transform(rescaleIMG)  # .astype(np.float32)
    # img2_scaled = (np.reshape(rescaleIMG, rgb.shape)).astype(np.uint8)
    # img2_scaled = img2_scaled.transpose((1, 2, 0))

    # normalized difference vegetation index (NDVI)
    ndvi = image.ndvi(rgbn)

    # Normalized Difference Water Index (NDWI)
    ndwi = image.ndwi(rgbn)

    # enhanced vegetation index (EVI)
    evi = image.evi(rgbn)

    # soil-adjusted vegetation index (SAVI)
    savi = image.savi(rgbn)

    # Iron Oxide Ratio (IOR)
    # This band ratio highlights hydrothermally
    # altered rocks that have been subjected to oxidation of iron-bearing sulphides.
    ior = image.ior(rgbn)

    # Burn Area Index (BAI)
    bai = image.bai(rgbn)

    # Leaf Area Index (LAI)
    lai = image.lai(rgbn)

    # Visible Atmospherically Resistant Index (VARI)
    vari = image.vari(rgbn)

    gemi = image.gemi(rgbn)

    gndvi = image.gndvi(rgbn)

    grvi = image.grvi(rgbn)

    sr = image.sr(rgbn)

    gs = gridspec.GridSpec(4, 4, right=0.9, left=0.1, hspace=0.2, wspace=0.1)

    plt.figure('jpg')
    plt.imshow(jpg)
    plt.figure()

    ax = plt.subplot(gs[0], title='red')
    ax.axis('off')
    ax.imshow(red, cmap="gray")
    ax.set_aspect('auto')

    ax = plt.subplot(gs[1], title='green')
    ax.axis('off')
    ax.imshow(green, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[2], title='blue')
    ax.axis('off')
    ax.imshow(blue, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[3], title='ndwi')
    ax.axis('off')
    ax.imshow(ndwi, cmap="gray")
    ax.set_aspect('auto')

    ax = plt.subplot(gs[4], title='ndvi')
    ax.axis('off')
    ax.imshow(ndvi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[5], title='evi')
    ax.axis('off')
    ax.imshow(evi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[6], title='savi')
    ax.axis('off')
    ax.imshow(savi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[7], title='ior')
    ax.axis('off')
    ax.imshow(ior, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[8], title='bai')
    ax.axis('off')
    ax.imshow(bai, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[9], title='vari')
    ax.axis('off')
    ax.imshow(vari, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[10], title='lai')
    ax.axis('off')
    ax.imshow(lai, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[11], title='gemi')
    ax.axis('off')
    ax.imshow(gemi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[12], title='gndvi')
    ax.axis('off')
    ax.imshow(gndvi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[13], title='grvi')
    ax.axis('off')
    ax.imshow(grvi, cmap='gray')
    ax.set_aspect('auto')

    ax = plt.subplot(gs[14], title='sr')
    ax.axis('off')
    ax.imshow(sr, cmap='gray')
    ax.set_aspect('auto')

    plt.show()

