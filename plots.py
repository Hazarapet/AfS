import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_curve(values, labels, file_name):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Steps')
    ax1.plot(values[0], 'r-', label=labels[0])
    ax1.set_ylabel('Loss', color='r')
    ax1.tick_params('y', colors='r')
    ax1.legend(loc=4)

    ax2 = ax1.twinx()

    for y_arr, label in zip(values[1:], labels[1:]):
        ax2.plot(y_arr, label=label)

    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1, 1), loc=5)
    plt.savefig(file_name, bbox_inches='tight')

def plot_augmented_train():
    df_train = pd.read_csv('train.csv')
    df_train_augmented = pd.read_csv('train-augmented.csv')
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    for label in labels:
        df_train[label] = df_train['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
        df_train_augmented[label] = df_train_augmented['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


