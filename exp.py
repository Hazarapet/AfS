import keras
import numpy as np
import keras.backend as K
from keras import losses
from utils import components
from keras.models import Sequential
from keras.layers import Dense

def reg_binary_cross_entropy(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred)

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=100))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=reg_binary_cross_entropy,
              metrics=['accuracy'])

# Generate dummy data
data = np.random.random((10, 100)).astype(np.int8)
labels = np.zeros((10, 3)).astype(np.int8)

for i in range(labels.shape[0]):
    r_int = np.random.randint(3)
    labels[i][r_int] = 1

print 'training...'
# Train the model, iterating on the data in batches of 32 samples
loss = model.fit(data, labels, epochs=2, batch_size=32, verbose=1)





