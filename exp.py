import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras.backend as K

def custom(y_true, y_pred):
    print 'custom: ', y_true, y_pred[1]
    return keras.objectives.binary_crossentropy(y_true, y_pred)

model = Sequential()
model.add(Dense(11, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=custom,
              metrics=['accuracy'])

# Generate dummy data
data = np.random.random((10, 100)).astype(np.int8)
labels = np.random.randint(2, size=(10, 1)).astype(np.int8)

print 'training...'
# Train the model, iterating on the data in batches of 32 samples
loss = model.fit(data, labels, nb_epoch=2, batch_size=32, verbose=1)





