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
from models.nm.model import model as nm_model
from models.nm.densenet121 import densenet121_model

st_time = time.time()
N_EPOCH = 10
BATCH_SIZE = 160
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
AUGMENT = True  # TODO somethings wrong with this.It also makes train slower

rare = ['conventional_mine', 'slash_burn', 'bare_ground', 'artisinal_mine',
        'blooming', 'selective_logging', 'blow_down']

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

df_tr = pd.read_csv('train_split.csv')
df_val = pd.read_csv('val_split.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

train, val = df_tr.values, df_val.values

print 'model loading...'
# [model, structure] = densenet121_model(
#     img_rows=IMAGE_WIDTH,
#     img_cols=IMAGE_HEIGHT,
#     color_type=3,
#     dropout_rate=0.35)

[model, structure] = nm_model('models/nm/structures/tr_l:0.1046-tr_a:0.9588-tr_f2:0.8941-val_l:0.1244-val_a:0.9535-val_f2:0.8787-time:26-06-2017-16:04:28-dur:42.68.h5')

print model.summary()

sgd = SGD(lr=1e-2, momentum=.9, decay=1e-4)

# model.compile(loss=components.f2_binary_cross_entropy(l=0.1),
#               optimizer=sgd,
#               metrics=[common_util.f2_score, 'accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=[common_util.f2_score, 'accuracy'])

print model.inputs
print "training..."

for epoch in range(N_EPOCH):
    tr_time = time.time()

    t_loss_graph_ep = []
    t_acc_graph_ep = []
    t_f2_graph_ep = []

    v_loss_graph_ep = []
    v_acc_graph_ep = []
    v_f2_graph_ep = []

    print "Epoch: {}".format(epoch)
    trained_batch = 0

    # we should shuffle the train set
    np.random.shuffle(train)

    for min_batch in common_util.iterate_minibatches(train, batchsize=BATCH_SIZE):

        t_batch_inputs = []
        t_batch_labels = []

        # now we should load min_batch's images and collect them
        for f, tags in min_batch:
            exists = False
            targets = np.zeros(17)

            for t in tags.split(' '):
                targets[label_map[t]] = 1
                if t in rare:
                    exists = True

            img = cv2.imread('resource/train-jpg/{}.jpg'.format(f))

            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float16)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))

            inputs = img

            t_batch_inputs.append(inputs)
            t_batch_labels.append(targets)

            if AUGMENT and exists:
                # --- augmentation ---
                t_batch_inputs = common_util.aug(t_batch_inputs, inputs)

                # cause 2x|input|
                t_batch_labels.append(targets)
                t_batch_labels.append(targets)

        t_batch_inputs = np.array(t_batch_inputs).astype(np.float16)
        t_batch_labels = np.array(t_batch_labels).astype(np.uint8)

        for min_b in common_util.iterate_minibatches(zip(t_batch_inputs, t_batch_labels), batchsize=BATCH_SIZE):
            # collecting for plotting
            t_i = np.stack(min_b[:, 0])  # inputs
            t_l = np.stack(min_b[:, 1])  # labels

            trained_batch += len(t_l)

            [t_loss, t_f2, t_acc] = model.train_on_batch(t_i, t_l)
            t_loss_graph_ep = np.append(t_loss_graph_ep, [t_loss])
            t_acc_graph_ep = np.append(t_acc_graph_ep, [t_acc])
            t_f2_graph_ep = np.append(t_f2_graph_ep, [t_f2])

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
            exists = False
            targets = np.zeros(17)

            for t in tags.split(' '):
                targets[label_map[t]] = 1
                if t in rare:
                    exists = True

            img = cv2.imread('resource/train-jpg/{}.jpg'.format(f))

            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float16)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))

            v_inputs = img

            v_batch_inputs.append(v_inputs)
            v_batch_labels.append(targets)

            if AUGMENT and exists:
                # --- augmentation ---
                v_batch_inputs = common_util.aug(v_batch_inputs, v_inputs)

                # cause 3x|input|
                v_batch_labels.append(targets)
                v_batch_labels.append(targets)
                v_batch_labels.append(targets)

        v_batch_inputs = np.array(v_batch_inputs).astype(np.float16)
        v_batch_labels = np.array(v_batch_labels).astype(np.uint8)

        [v_loss, v_f2, v_acc] = model.evaluate(v_batch_inputs, v_batch_labels, batch_size=BATCH_SIZE, verbose=0)

        val_batch += len(min_batch)

        v_loss_graph_ep = np.append(v_loss_graph_ep, [v_loss])
        v_acc_graph_ep = np.append(v_acc_graph_ep, [v_acc])
        v_f2_graph_ep = np.append(v_f2_graph_ep, [v_f2])

        print "Val Examples: {}/{}, loss: {:.5f}, acc: {:.5f}, f2: {:.5f}, l_rate: {:.5f} | {:.1f}m".format(val_batch,
            len(val),
            float(v_loss),
            float(v_acc),
            float(v_f2),
            float(model.optimizer.lr.get_value()),
            (time.time() - tr_time) / 60)

        # if model has reach to good results, we save that model
        if v_f2 > 0.905:
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

    if epoch == 8:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-3)

    if epoch == 20:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-3)

    if epoch == 30:
        lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(1e-4)

    t_loss_graph = np.append(t_loss_graph, [np.mean(t_loss_graph_ep)])
    t_acc_graph = np.append(t_acc_graph, [np.mean(t_acc_graph_ep)])
    t_f2_graph = np.append(t_f2_graph, [np.mean(t_f2_graph_ep)])

    v_loss_graph = np.append(v_loss_graph, [np.mean(v_loss_graph_ep)])
    v_acc_graph = np.append(v_acc_graph, [np.mean(v_acc_graph_ep)])
    v_f2_graph = np.append(v_f2_graph, [np.mean(v_f2_graph_ep)])


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
