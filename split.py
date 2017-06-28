import numpy as np
import pandas as pd

df_train = pd.read_csv('train_v2.csv')

# we should shuffle all examples
np.random.shuffle(df_train.values)

# splitting to train and validation set
index = int(len(df_train.values) * 0.95)
train, val = df_train.values[:index], df_train.values[index:]

df_tr = pd.DataFrame([[f, t] for f, t in train])
df_tr.columns = ['image_name', 'tags']

df_tr.to_csv('train_split.csv', index=False)

df_val = pd.DataFrame([[f, t] for f, t in val])
df_val.columns = ['image_name', 'tags']

df_val.to_csv('val_split.csv', index=False)

