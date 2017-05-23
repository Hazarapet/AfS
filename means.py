import cv2
import sys
import time
import json
import plots
import numpy as np
import pandas as pd
from utils import image as UtilImage

IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

df_train = pd.read_csv('train_v2.csv')

total_red = total_green = total_blue = total_nir = total_ndvi = total_ndwi = total_ior = total_bai = total_gemi = 0.
total = 0

for f, tags in df_train.values:
    rgbn = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
    assert rgbn is not None

    ndvi = UtilImage.ndvi(rgbn)
    ndwi = UtilImage.ndwi(rgbn)
    ior = UtilImage.ior(rgbn)
    bai = UtilImage.bai(rgbn)
    gemi = UtilImage.gemi(rgbn)

    # resize
    red = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGH))
    green = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGH))
    blue = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGH))
    nir = cv2.resize(rgbn[3], (IMAGE_WIDTH, IMAGE_HEIGH))
    ndvi = cv2.resize(ndvi, (IMAGE_WIDTH, IMAGE_HEIGH))
    ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGH))
    ior = cv2.resize(ior, (IMAGE_WIDTH, IMAGE_HEIGH))
    bai = cv2.resize(bai, (IMAGE_WIDTH, IMAGE_HEIGH))
    gemi = cv2.resize(gemi, (IMAGE_WIDTH, IMAGE_HEIGH))

    total_red += np.sum(red)
    total_green += np.sum(green)
    total_blue += np.sum(blue)
    total_nir += np.sum(nir)
    total_ndvi += np.sum(ndvi)
    total_ndwi += np.sum(ndwi)
    total_ior += np.sum(ior)
    total_bai += np.sum(bai)
    total_gemi += np.sum(gemi)

    print "{}/{}".format(total, len(df_train.values))
    total += 1

total_count = len(df_train.values)

mean_red = total_red / total_count
mean_green = total_green / total_count
mean_blue = total_blue / total_count
mean_nir = total_nir / total_count
mean_ndvi = total_ndvi / total_count
mean_ndwi = total_ndwi / total_count
mean_ior = total_ior / total_count
mean_bai = total_bai / total_count
mean_gemi = total_gemi / total_count

bands = ['red', 'green', 'blue', 'nir', 'ndvi', 'ndwi', 'ior', 'bai', 'gemi']
means = [mean_red, mean_green, mean_blue, mean_nir, mean_ndvi, mean_ndwi, mean_ior, mean_bai, mean_gemi]

df_means = pd.DataFrame([[b, m] for b, m in zip(bands, means)])
df_means.columns = ['band', 'mean']

df_means.to_csv('means.csv', index=False)
