import cv2
import sys
import time
import json
import plots
import numpy as np
import pandas as pd
from utils import image as UtilImage
from utils import components
from keras.optimizers import SGD, Adam
from utils import common as common_util
from models.A.model import model as A_model
from models.UNET.model import model as unet_model

st_time = time.time()
N_EPOCH = 7
BATCH_SIZE = 80
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

t_loss_graph = np.array([])
t_acc_graph = np.array([])
v_loss_graph = np.array([])
v_acc_graph = np.array([])

X = []
y = []

print 'data loading...'
# loading the data
df_train = pd.read_csv('train-augmented.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# we should shuffle all examples
np.random.shuffle(df_train.values)

# splitting to train and validation set
index = int(len(df_train.values) * 0.8)
train, val = df_train.values[:index], df_train.values[index:]

print 'model loading...'
[model, structure] = unet_model('models/UNET/structures/tr_l:0.049-tr_a:0.987-val_l:0.116-val_a:0.96-time:11-05-2017-02:39:02-dur:640.645.h5')

adam = Adam(lr=1e-5, decay=0.)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print model.inputs
print "training..."

for epoch in range(N_EPOCH):

    print "Epoch: {}".format(epoch)
    trained_batch = 0

    # we should shuffle the train set
    np.random.shuffle(train)

    for min_batch in common_util.iterate_minibatches(train, batchsize=BATCH_SIZE):

        t_batch_inputs = []
        t_batch_sobel_inputs = []
        t_batch_labels = []

        # accumulate the examples' count
        trained_batch += len(min_batch)

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            img = cv2.imread('resource/train-augmented-jpg/{}.jpg'.format(f))
            assert img is not None

            if img is not None:
                targets = np.zeros(17)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1

                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGH)).astype(np.float32)
                img[:, :, 0] -= 103.939
                img[:, :, 1] -= 116.779
                img[:, :, 2] -= 123.68
                img = img.transpose((2, 0, 1))

                # mag, angle = UtilImage.img_sobel(img)

                t_batch_inputs.append(img)
                # t_batch_sobel_inputs.append(mag)
                t_batch_labels.append(targets)

        # t_batch_sobel_inputs = np.array(t_batch_sobel_inputs).astype(np.float16)
        t_batch_inputs = np.array(t_batch_inputs).astype(np.float16)
        t_batch_labels = np.array(t_batch_labels).astype(np.int8)

        # collecting for plotting
        [t_loss, t_acc] = model.train_on_batch(t_batch_inputs, t_batch_labels)
        t_loss_graph = np.append(t_loss_graph, [t_loss])
        t_acc_graph = np.append(t_acc_graph, [t_acc])

        print "examples: {}/{}, loss: {:.4f}, accuracy: {:.3f}".format(trained_batch,
               len(train),
               float(t_loss),
               float(t_acc))

    # ===== Validation =====
    v_batch_inputs = []
    # v_batch_sobel_inputs = []
    v_batch_labels = []

    np.random.shuffle(val)

    # load val's images
    for f, tags in val:
        img = cv2.imread('resource/train-augmented-jpg/{}.jpg'.format(f))
        assert img is not None

        if img is not None:
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1

            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGH)).astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))

            # mag, angle = UtilImage.img_sobel(img)

            v_batch_inputs.append(img)
            # v_batch_sobel_inputs.append(mag)
            v_batch_labels.append(targets)

    # v_batch_sobel_inputs = np.array(v_batch_sobel_inputs).astype(np.float16)
    v_batch_inputs = np.array(v_batch_inputs).astype(np.float16)
    v_batch_labels = np.array(v_batch_labels).astype(np.int8)

    [v_loss, v_acc] = model.evaluate(v_batch_inputs, v_batch_labels, batch_size=BATCH_SIZE)
    v_loss_graph = np.append(v_loss_graph, [v_loss])
    v_acc_graph = np.append(v_acc_graph, [v_acc])

    if epoch == 15:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-5)

    if epoch == 20:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-5)

    # if model has reach to good results, we save that model
    if v_loss < 0.07:
        timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
        model_filename = structure + 'good-epoch:' + str(epoch) + \
                         '-tr_l:' + str(round(np.min(t_loss_graph), 3)) + \
                         '-tr_a:' + str(round(np.max(t_acc_graph), 3)) + \
                         '-val_l:' + str(round(v_loss, 3)) + \
                         '-val_a:' + str(round(v_acc, 3)) + \
                         '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))
        # saving the weights
        model.save_weights(model_filename + '.h5')

        with open(model_filename + '.json', 'w') as outfile:
            json_string = model.to_json()
            json.dump(json_string, outfile)

    print "Val Examples: {}, loss: {:.4f}, accuracy: {:.3f}, l_rate: {:.5f}".format(
        len(val),
        float(v_loss),
        float(v_acc),
        float(model.optimizer.lr.get_value()))

# create file name to save the state with useful information
timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
model_filename = structure + \
                 'tr_l:' + str(round(np.min(t_loss_graph), 3)) + \
                 '-tr_a:' + str(round(np.max(t_acc_graph), 3)) + \
                 '-val_l:' + str(round(np.min(v_loss_graph), 3)) + \
                 '-val_a:' + str(round(np.max(v_acc_graph), 3)) + \
                 '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))

# saving the weights
model.save_weights(model_filename + '.h5')

with open(model_filename + '.json', 'w') as outfile:
    json_string = model.to_json()
    json.dump(json_string, outfile)

# --------------------------------------
# --------- Plotting Curves -----------
plots.plot_curve(values=t_loss_graph, title='Training Loss', file_name=model_filename + '_tr_loss.jpg')
plots.plot_curve(values=t_acc_graph, title='Training Accuracy', file_name=model_filename + '_tr_acc.jpg', y_axis='Accuracy')
plots.plot_curve(values=v_loss_graph, title='Val Loss', file_name=model_filename + '_val_loss.jpg')
plots.plot_curve(values=v_acc_graph, title='Val Accuracy', file_name=model_filename + '_val_acc.jpg', y_axis='Accuracy')

print 'Loss and Accuracy plots are done!'

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'
