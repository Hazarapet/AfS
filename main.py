import cv2
import time
import json
import plots
import numpy as np
import pandas as pd
from keras.optimizers import SGD, Adam
from utils import common as common_util
from models.A.model import model as A_model

st_time = time.time()
N_EPOCH = 5
BATCH_SIZE = 80
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

t_loss_graph = np.array([])
t_acc_graph = np.array([])
v_loss_graph = np.array([])
v_acc_graph = np.array([])

x_train = []
y_train = []

print 'data loading...'
# loading the data
df_train = pd.read_csv('train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}


# for f, tags in df_train.values[:20000]:
#     img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
#     targets = np.zeros(17)
#     for t in tags.split(' '):
#         targets[label_map[t]] = 1
#     x_train.append(cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGH)))
#     y_train.append(targets)

print 'model loading...'
[model, structure] = A_model()

adam = Adam(lr=0.0001, decay=0.)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print model.inputs
print "training..."

for epoch in range(N_EPOCH):

    print "Epoch: {}".format(epoch)
    trained_batch = 0

    # we should shuffle the train set
    [x_train, y_train] = common_util.parallel_shuffle(x_train, y_train)

    for [x_batch, y_batch] in common_util.iterate_minibatches([x_train, y_train]):
        # accumulate the examples' count
        trained_batch += len(x_batch)

        # collecting for plotting
        [t_loss, t_acc] = model.train_on_batch(x_batch, y_batch)
        t_loss_graph = np.append(t_loss_graph, [t_loss])
        t_acc_graph = np.append(t_acc_graph, [t_acc])

        print "examples: {}/{}, loss: {:.4f}, accuracy: {:.3f}".format(trained_batch,
               len(x_train),
               float(t_loss),
               float(t_acc))

    # ===== Validation =====
    [v_loss, v_acc] = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
    v_loss_graph = np.append(v_loss_graph, [v_loss])
    v_acc_graph = np.append(v_acc_graph, [v_acc])

    print "Val Examples: {}, loss: {:.4f}, accuracy: {:.3f}, l_rate: {:.5f}".format(
        len(x_train),
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
plots.plot_curve(values=t_acc_graph, title='Val Accuracy', file_name=model_filename + '_val_acc.jpg', y_axis='Accuracy')

print 'Loss and Accuracy plots are ready'

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'
