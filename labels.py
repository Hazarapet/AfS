import pandas as pd

df_train = pd.read_csv('train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

df_labels = pd.DataFrame([[i, l] for i, l in enumerate(labels)])
df_labels.columns = ['class', 'label']

df_labels.to_csv('labels.csv', index=False)
