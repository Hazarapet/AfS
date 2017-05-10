import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_curve(values, title, file_name, x_axis='Step', y_axis='Loss'):
    plt.figure(title)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel(y_axis, fontsize=12)
    plt.plot(values, '-b', label=title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
    plt.savefig(file_name, bbox_inches='tight')

def plot_augmented_train():
    df_train = pd.read_csv('train.csv')
    df_train_augmented = pd.read_csv('train-augmented.csv')
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    for label in labels:
        df_train[label] = df_train['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
        df_train_augmented[label] = df_train_augmented['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


