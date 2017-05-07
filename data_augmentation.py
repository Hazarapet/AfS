import os
import cv2
import shutil
import numpy as np
import pandas as pd
from PIL import Image

# if horizontal:
#     im = im.transpose(Image.FLIP_LEFT_RIGHT)
#
# if vertical:
#     im = im.transpose(Image.FLIP_TOP_BOTTOM)

COMMON_TAGS = ['clear', 'primary', 'agriculture']


def create_folder(dir_name=None):
    if os.path.exists(dir_name) and dir_name:
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)

def image_data_augmentation(horizontal_flip=False, vertical_flip=False, rotation=0):
    df_train = pd.read_csv('train.csv')

    tags = df_train['tags'].values
    images = df_train['image_name'].values
    img_dest_name = 'resource/train-augmented-jpg/'

    create_folder(img_dest_name)
    count = 0

    for f, t in df_train.values:
        img = cv2.imread('resource/train-jpg/{}.jpg'.format(f))
        assert img is not None

        cv2.imwrite(img_dest_name + f + '.jpg', img)

        if len(list(set(t.split(' ')) & set(COMMON_TAGS))) == 0:
            img = Image.open('resource/train-jpg/{}.jpg'.format(f))

            if horizontal_flip:
                img_h = img.transpose(Image.FLIP_LEFT_RIGHT)
                tags = np.append(tags, t)
                images = np.append(images, 'h_flip_' + f)
                img_h.save(img_dest_name + 'h_flip_' + f + '.jpg')

            if vertical_flip:
                img_v = img.transpose(Image.FLIP_TOP_BOTTOM)
                tags = np.append(tags, t)
                images = np.append(images, 'v_flip_' + f)
                img_v.save(img_dest_name + 'v_flip_' + f + '.jpg')

            if rotation:
                img_rt = img.rotate(rotation)
                tags = np.append(tags, t)
                images = np.append(images, 'r_' + str(rotation) + '_' + f)
                img_rt.save(img_dest_name + 'r_' + str(rotation) + '_' + f + '.jpg')

        print '{}/{} augmented'.format(str(count), str(len(df_train.values)))
        count += 1

    df_a_train = pd.DataFrame([[im, tags[i]] for i, im in enumerate(images)])
    df_a_train.columns = ['image_name', 'tags']
    df_a_train.to_csv('train-augmented.csv', index=False)


if __name__ == '__main__':
    image_data_augmentation(horizontal_flip=True)
