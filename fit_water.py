import cv2
import sys
import time
import json
import plots
import numpy as np
import pandas as pd
from utils import components
from keras.optimizers import Adam, SGD
from utils import image as UtilImage
from utils import common as common_util
from models.water.resnet50 import model as resnet_model

st_time = time.time()
N_EPOCH = 15
BATCH_SIZE = 100
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
AUGMENT_SCALE = 5

AUGMENT = True  # TODO somethings wrong with this.It also makes train slower

rare = ['conventional_mine', 'slash_burn', 'bare_ground', 'artisinal_mine',
        'blooming', 'selective_logging', 'blow_down', 'cultivation', 'road', 'habitation', 'water']

t_loss_graph = np.array([])
t_f2_graph = np.array([])
v_loss_graph = np.array([])
v_f2_graph = np.array([])

X = []
y = []

print 'data loading...'
# loading the data
df_train = pd.read_csv('train_v2.csv')

df_tr = pd.read_csv('train_split.csv')
df_val = pd.read_csv('val_split.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# TODO Whole db for training
train, val = df_tr.values, df_val.values

print 'model loading...'
[model, structure] = resnet_model()

print model.summary()

sgd = SGD(lr=5e-2, momentum=.9, decay=1e-6, nesterov=True)

# model.compile(loss=components.f2_binary_cross_entropy(l=1e-1),
#               optimizer=sgd,
#               metrics=[common_util.f2_score])
#
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=[common_util.f2_score])

print model.inputs
print "training..."

for epoch in range(N_EPOCH):
    tr_time = time.time()

    t_loss_graph_ep = []
    t_f2_graph_ep = []

    v_loss_graph_ep = []
    v_f2_graph_ep = []

    print "Epoch: {}".format(epoch)
    trained_batch = 0

    # we should shuffle the train set
    np.random.shuffle(train)

    for min_batch in common_util.iterate_minibatches(train, batchsize=BATCH_SIZE):

        t_batch_inputs256 = []
        t_batch_labels = []

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            exists = False  # TODO to augment all examples
            targets = 0

            for t in tags.split(' '):
                if t == 'water':
                    targets = 1
                    exists = True

            img = cv2.imread('resource/train-jpg/{}.jpg'.format(f))
            img = img / 256.

            rgbn = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
            ndvi = UtilImage.ndvi(rgbn)

            ndvi = cv2.resize(ndvi, (256, 256)).astype(np.float32)

            img256 = cv2.resize(img, (256, 256)).astype(np.float32)
            img256 = img256.transpose((2, 0, 1))

            inputs256 = np.append(img256[1:], [ndvi], axis=0)

            t_batch_inputs256.append(inputs256)

            t_batch_labels.append(targets)

            if AUGMENT and exists:
                # --- augmentation ---
                t_batch_inputs256 = common_util.aug(t_batch_inputs256, inputs256)

                # cause AUGMENT_SCALEx|input|
                for i in range(AUGMENT_SCALE):
                    t_batch_labels.append(targets)

        t_batch_inputs256 = np.array(t_batch_inputs256).astype(np.float32)

        t_batch_labels = np.array(t_batch_labels).astype(np.uint8)

        rn = [[i] for i in range(t_batch_labels.shape[0])]

        for min_b in common_util.iterate_minibatches(zip(rn, t_batch_labels), batchsize=BATCH_SIZE):
            indices = np.stack(min_b[:, 0])  # inputs
            indices = indices.reshape(indices.shape[0])  # inputs
            t_i = t_batch_inputs256[indices]
            t_l = np.stack(min_b[:, 1])     # labels

            trained_batch += len(t_l)

            [t_loss, t_f2] = model.train_on_batch(t_i, t_l)
            t_loss_graph_ep = np.append(t_loss_graph_ep, [t_loss])
            t_f2_graph_ep = np.append(t_f2_graph_ep, [t_f2])

            print "examples: {}/{}/{}, loss: {:.5f}, f2: {:.5f}".format(trained_batch,
                   len(train) * (AUGMENT_SCALE + 1),
                   len(train),
                   float(t_loss),
                   float(t_f2))

    # ===== Validation =====
    print '----- Validation of epoch: {} -----'.format(epoch)
    np.random.shuffle(val)
    val_batch = 0
    for min_batch in common_util.iterate_minibatches(val, batchsize=128):

        v_batch_inputs256 = []
        v_batch_labels = []

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            exists = False  # TODO to augment all examples
            targets = 0

            for t in tags.split(' '):
                if t == 'water':
                    targets = 1
                    exists = True

            img = cv2.imread('resource/train-jpg/{}.jpg'.format(f))
            img = img / 256.

            rgbn = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
            ndvi = UtilImage.ndvi(rgbn)

            ndvi = cv2.resize(ndvi, (256, 256)).astype(np.float32)

            img256 = cv2.resize(img, (256, 256)).astype(np.float32)
            img256 = img256.transpose((2, 0, 1))

            v_inputs256 = np.append(img256[1:], [ndvi], axis=0)

            v_batch_inputs256.append(v_inputs256)

            v_batch_labels.append(targets)

            if AUGMENT and exists:
                # --- augmentation ---
                v_batch_inputs256 = common_util.aug(v_batch_inputs256, v_inputs256)

                # cause AUGMENT_SCALEx|input|
                for i in range(AUGMENT_SCALE):
                    v_batch_labels.append(targets)

        v_batch_inputs256 = np.array(v_batch_inputs256).astype(np.float32)

        v_batch_labels = np.array(v_batch_labels).astype(np.uint8)

        v_batch_inputs = v_batch_inputs256

        [v_loss, v_f2] = model.evaluate(v_batch_inputs, v_batch_labels, batch_size=BATCH_SIZE, verbose=0)

        val_batch += len(min_batch)

        v_loss_graph_ep = np.append(v_loss_graph_ep, [v_loss])
        v_f2_graph_ep = np.append(v_f2_graph_ep, [v_f2])

    # if model has reach to good results, we save that model
    if np.mean(v_f2_graph_ep) > 0.95:
        timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
        model_filename = structure + 'good-epoch:' + str(epoch) + \
                         '-tr_l:' + str(round(np.mean(t_loss_graph_ep), 4)) + \
                         '-tr_f2:' + str(round(np.mean(t_f2_graph_ep), 4)) + \
                         '-val_l:' + str(round(np.mean(v_loss_graph_ep), 4)) + \
                         '-val_f2:' + str(round(np.mean(v_f2_graph_ep), 4)) + \
                         '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))
        # saving the weights
        model.save_weights(model_filename + '.h5')

        with open(model_filename + '.json', 'w') as outfile:
            json_string = model.to_json()
            json.dump(json_string, outfile)

    print "Val Examples: {}/{}/{}, loss: {:.5f}, f2: {:.5f}, l_rate: {:.5f} | {:.1f}m".format(val_batch,
       len(val) * (AUGMENT_SCALE + 1),
       len(val),
       float(np.mean(v_loss_graph_ep)),
       float(np.mean(v_f2_graph_ep)),
       float(model.optimizer.lr.get_value()),
       (time.time() - tr_time) / 60)

    if epoch == 5:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-2)

    if epoch == 10:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-3)

    if epoch == 15:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(3e-4)

    t_loss_graph = np.append(t_loss_graph, [np.mean(t_loss_graph_ep)])
    t_f2_graph = np.append(t_f2_graph, [np.mean(t_f2_graph_ep)])

    v_loss_graph = np.append(v_loss_graph, [np.mean(v_loss_graph_ep)])
    v_f2_graph = np.append(v_f2_graph, [np.mean(v_f2_graph_ep)])


# create file name to save the state with useful information
timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
model_filename = structure + \
                 'tr_l:' + str(round(np.min(t_loss_graph), 4)) + \
                 '-tr_f2:' + str(round(np.max(t_f2_graph), 4)) + \
                 '-val_l:' + str(round(np.min(v_loss_graph), 4)) + \
                 '-val_f2:' + str(round(np.max(v_f2_graph), 4)) + \
                 '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))

# saving the weights
model.save_weights(model_filename + '.h5')

with open(model_filename + '.json', 'w') as outfile:
    json_string = model.to_json()
    json.dump(json_string, outfile)

# --------------------------------------
# --------- Plotting Curves ------------
# train, val, train, val etc...
plots.plot_curve(values=[t_loss_graph, v_loss_graph, t_f2_graph, v_f2_graph], labels=['Train Loss', 'Val Loss', 'Train F2', 'Val F2'], file_name=model_filename + '_plot.jpg')

print 'Loss and F2 plots are done!'

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'
