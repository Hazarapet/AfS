import cv2
import sys
import time
import json
import plots
import numpy as np
import pandas as pd
from utils import components
from keras.optimizers import Adam
from utils import image as UtilImage
from utils import common as common_util
from models.nm.model import model as nm_model

st_time = time.time()
N_EPOCH = 10
BATCH_SIZE = 100
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

t_loss_graph = np.array([])
t_acc_graph = np.array([])
t_f2_graph = np.array([])
v_loss_graph = np.array([])
v_acc_graph = np.array([])
v_f2_graph = np.array([])

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
index = int(len(df_train.values) * 0.85)
train, val = df_train.values[:index], df_train.values[index:]

print 'model loading...'
[model, structure] = nm_model()

print model.summary()
sys.exit(0)

adam = Adam(lr=6e-3, decay=0.)

model.compile(loss=components.f2_binary_cross_entropy(),
              optimizer=adam,
              metrics=[common_util.f2_score, 'accuracy'])

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

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            rgbn = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
            assert rgbn is not None

            if rgbn is not None:
                targets = np.zeros(17)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1

                ndvi = UtilImage.ndvi(rgbn)
                ndwi = UtilImage.ndwi(rgbn)
                ior = UtilImage.ior(rgbn)
                bai = UtilImage.bai(rgbn)

                # resize
                red = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGHT))
                green = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGHT))
                blue = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGHT))
                nir = cv2.resize(rgbn[3], (IMAGE_WIDTH, IMAGE_HEIGHT))
                ndvi = cv2.resize(ndvi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                ior = cv2.resize(ior, (IMAGE_WIDTH, IMAGE_HEIGHT))
                bai = cv2.resize(bai, (IMAGE_WIDTH, IMAGE_HEIGHT))

                # red, green, blue, nir, ndvi, ndwi, ior, bai
                inputs = [red, green, blue, nir, ndvi, ndwi, ior, bai]

                t_batch_inputs.append(inputs)
                t_batch_labels.append(targets)

                if True:
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
                    flip_h_inputs = np.flip(inputs, 2)
                    t_batch_inputs.append(flip_h_inputs)
                    t_batch_labels.append(targets)

                    # flip v
                    flip_v_inputs = np.flip(inputs, 1)
                    t_batch_inputs.append(flip_v_inputs)
                    t_batch_labels.append(targets)

        t_batch_inputs = np.array(t_batch_inputs).astype(np.float32)
        t_batch_labels = np.array(t_batch_labels).astype(np.uint8)

        for min_b in common_util.iterate_minibatches(zip(t_batch_inputs, t_batch_labels), batchsize=BATCH_SIZE):
            # collecting for plotting
            t_i = np.stack(min_b[:, 0])  # inputs
            t_l = np.stack(min_b[:, 1])  # labels

            trained_batch += len(t_l)

            [t_loss, t_f2, t_acc] = model.train_on_batch(t_i, t_l)
            t_loss_graph = np.append(t_loss_graph, [t_loss])
            t_acc_graph = np.append(t_acc_graph, [t_acc])
            t_f2_graph = np.append(t_f2_graph, [t_f2])

            print "examples: {}/{}, loss: {:.5f}, acc: {:.5f}, f2: {:.5f}".format(trained_batch,
                   len(train),
                   float(t_loss),
                   float(t_acc),
                   float(t_f2))

    # ===== Validation =====
    print '----- Validation of epoch: {} -----'.format(epoch)
    np.random.shuffle(val)
    val_batch = 0
    for min_batch in common_util.iterate_minibatches(val, batchsize=1024):
        v_batch_inputs = []
        v_batch_labels = []

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            rgbn = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
            assert rgbn is not None

            if rgbn is not None:
                targets = np.zeros(17)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1

                ndvi = UtilImage.ndvi(rgbn)
                ndwi = UtilImage.ndwi(rgbn)
                ior = UtilImage.ior(rgbn)
                bai = UtilImage.bai(rgbn)

                # resize
                red = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGHT))
                green = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGHT))
                blue = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGHT))
                nir = cv2.resize(rgbn[3], (IMAGE_WIDTH, IMAGE_HEIGHT))
                ndvi = cv2.resize(ndvi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                ior = cv2.resize(ior, (IMAGE_WIDTH, IMAGE_HEIGHT))
                bai = cv2.resize(bai, (IMAGE_WIDTH, IMAGE_HEIGHT))

                # red, green, blue, nir, ndvi, ndwi, ior, bai
                v_inputs = [red, green, blue, nir, ndvi, ndwi, ior, bai]

                v_batch_inputs.append(v_inputs)
                v_batch_labels.append(targets)

                if True:
                    # --- augmentation ---
                    # rotate 90
                    rt90_inputs = np.rot90(v_inputs, 1, axes=(1, 2))
                    v_batch_inputs.append(rt90_inputs)
                    v_batch_labels.append(targets)

                    # rotate 180
                    rt180_inputs = np.rot90(v_inputs, 2, axes=(1, 2))
                    v_batch_inputs.append(rt180_inputs)
                    v_batch_labels.append(targets)

                    # flip h
                    flip_h_inputs = np.flip(v_inputs, 2)
                    v_batch_inputs.append(flip_h_inputs)
                    v_batch_labels.append(targets)

                    # flip v
                    flip_v_inputs = np.flip(v_inputs, 1)
                    v_batch_inputs.append(flip_v_inputs)
                    v_batch_labels.append(targets)

        v_batch_inputs = np.array(v_batch_inputs).astype(np.float32)
        v_batch_labels = np.array(v_batch_labels).astype(np.uint8)

        [v_loss, v_f2, v_acc] = model.evaluate(v_batch_inputs, v_batch_labels, batch_size=BATCH_SIZE, verbose=0)

        val_batch += len(min_batch)

        v_loss_graph = np.append(v_loss_graph, [v_loss])
        v_f2_graph = np.append(v_f2_graph, [v_f2])
        v_acc_graph = np.append(v_acc_graph, [v_acc])

        print "Val Examples: {}/{}, loss: {:.5f}, acc: {:.5f}, f2: {:.5f}, l_rate: {:.5f}".format(val_batch,
            len(val),
            float(v_loss),
            float(v_acc),
            float(v_f2),
            float(model.optimizer.lr.get_value()))

        # if model has reach to good results, we save that model
        if v_f2 > 0.8:
            timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
            model_filename = structure + 'good-epoch:' + str(epoch) + \
                             '-tr_l:' + str(round(np.min(t_loss_graph), 4)) + \
                             '-tr_a:' + str(round(np.max(t_acc_graph), 4)) + \
                             '-tr_f2:' + str(round(np.max(t_f2_graph), 4)) + \
                             '-val_l:' + str(round(v_loss, 4)) + \
                             '-val_a:' + str(round(np.max(v_acc_graph), 4)) + \
                             '-val_f2:' + str(round(np.max(v_f2_graph), 4)) + \
                             '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))
            # saving the weights
            model.save_weights(model_filename + '.h5')

            with open(model_filename + '.json', 'w') as outfile:
                json_string = model.to_json()
                json.dump(json_string, outfile)

    if epoch == 5:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(3e-3)

    if epoch == 8:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-3)


# create file name to save the state with useful information
timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
model_filename = structure + \
                 'tr_l:' + str(round(np.min(t_loss_graph), 4)) + \
                 '-tr_a:' + str(round(np.max(t_acc_graph), 4)) + \
                 '-tr_f2:' + str(round(np.max(t_f2_graph), 4)) + \
                 '-val_l:' + str(round(np.min(v_loss_graph), 4)) + \
                 '-val_a:' + str(round(np.max(v_acc_graph), 4)) + \
                 '-val_f2:' + str(round(np.max(v_f2_graph), 4)) + \
                 '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))

# saving the weights
model.save_weights(model_filename + '.h5')

with open(model_filename + '.json', 'w') as outfile:
    json_string = model.to_json()
    json.dump(json_string, outfile)

# --------------------------------------
# --------- Plotting Curves -----------
plots.plot_curve(values=[t_loss_graph, t_acc_graph, t_f2_graph], labels=['Train Loss', 'Train Acc', 'Train F2'], file_name=model_filename + '_train.jpg')
plots.plot_curve(values=[v_loss_graph, v_acc_graph, v_f2_graph], labels=['Val Loss', 'Val Acc', 'Val F2'], file_name=model_filename + '_val.jpg')

print 'Loss and Accuracy plots are done!'

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'