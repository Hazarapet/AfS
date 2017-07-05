import numpy as np


def iterate_minibatches(inputs, batchsize=10):

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt]

    if len(inputs) % batchsize != 0:
        yield np.array(inputs)[- (len(inputs) % batchsize):]

t_batch_labels = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
rn = np.zeros((t_batch_labels.shape[0], 1))
rnn = np.arange(t_batch_labels.shape[0])
rnnn = [[i] for i in range(t_batch_labels.shape[0])]

arr = iterate_minibatches(zip(rnnn, t_batch_labels), batchsize=2)

print rn
print rnn
print rnnn
for a in arr:
    print a