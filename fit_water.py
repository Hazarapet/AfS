import cv2
import sys
import time
import json
import plots
import numpy as np
import pandas as pd
import tif_data_augmentation as tfa
from utils import components
from utils import image as UtilImage
from keras.optimizers import SGD, Adam
from utils import common as common_util
from models.water.model import model as water_model

st_time = time.time()
N_EPOCH = 10
BATCH_SIZE = 10
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

t_loss_graph = np.array([])
t_acc_graph = np.array([])
v_loss_graph = np.array([])
v_acc_graph = np.array([])
v_predict = np.array([])

X = []
y = []

print 'data loading...'
# loading the data
df_train = pd.read_csv('train_v2.csv')

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
[model, structure] = water_model()

adam = Adam(lr=1e-4, decay=0.)

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
        t_batch_labels = []

        # accumulate the examples' count
        trained_batch += len(min_batch)

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            rgbn, ndwi, _, _, _ = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
            assert rgbn is not None

            if rgbn is not None:
                targets = 0
                for t in tags.split(' '):
                    if t == 'water':
                        targets = 1

                # resize
                r = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGH))
                g = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGH))
                b = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGH))
                ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGH))

                inputs = [r, g, b, ndwi]

                t_batch_inputs.append(inputs)
                t_batch_labels.append(targets)

                if targets == 1:
                    # --- augmentation ---
                    # rotate 90
                    rt90_inputs = np.rot90(inputs, 1, axes=(1, 2))
                    t_batch_inputs.append(rt90_inputs)
                    t_batch_labels.append(targets)

                    # rotate 180
                    rt180_inputs = np.rot90(inputs, 2, axes=(1, 2))
                    t_batch_inputs.append(rt180_inputs)
                    t_batch_labels.append(targets)

                    # flip h
                    flip_h_inputs = np.fliplr(inputs)
                    t_batch_inputs.append(flip_h_inputs)
                    t_batch_labels.append(targets)

                    # flip v
                    flip_v_inputs = np.flipud(inputs)
                    t_batch_inputs.append(flip_v_inputs)
                    t_batch_labels.append(targets)

        t_batch_inputs = np.array(t_batch_inputs).astype(np.float32)
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
    np.random.shuffle(val)
    # v_labels = []

    for min_batch in common_util.iterate_minibatches(val, batchsize=BATCH_SIZE):
        v_batch_inputs = []
        v_batch_labels = []

        # load val's images
        for f, tags in min_batch:
            rgbn, ndwi, _, _, _ = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
            assert rgbn is not None

            if rgbn is not None:
                targets = 0
                for t in tags.split(' '):
                    if t == 'water':
                        targets = 1

                # resize
                r = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGH))
                g = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGH))
                b = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGH))
                ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGH))

                v_batch_inputs.append([r, g, b, ndwi])
                v_batch_labels.append(targets)
                # v_labels.append(targets)

        v_batch_inputs = np.array(v_batch_inputs).astype(np.float32)
        v_batch_labels = np.array(v_batch_labels).astype(np.int8)

        [v_loss, v_acc] = model.evaluate(v_batch_inputs, v_batch_labels, batch_size=BATCH_SIZE, verbose=0)
        # [v_p] = model.predict_on_batch(v_batch_inputs)

        v_loss_graph = np.append(v_loss_graph, [v_loss])
        v_acc_graph = np.append(v_acc_graph, [v_acc])
        # v_predict = np.append(v_predict, [v_p])

    if epoch == 15:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-5)

    if epoch == 20:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-5)

    # if model has reach to good results, we save that model
    if v_loss < 0.0002:
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

    # v_labels = np.array(v_labels).astype(np.uint8)

    print "Val Examples: {}, loss: {:.5f}, accuracy: {:.5f}, f2: {:.5f}, l_rate: {:.5f}".format(
        len(val),
        float(v_loss),
        float(v_acc),
        # float(common_util.f2_score(v_labels, v_predict > .5)),
        float(77.77),
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
