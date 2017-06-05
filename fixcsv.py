import pandas as pd

df_test = pd.read_csv('submission_0.csv')

names = []

for f in df_test['image_name'].values:
    names.append(f.replace('.tif', ''))

df_test['image_name'] = names
df_test.to_csv('submission_fixed.csv', index=False)
