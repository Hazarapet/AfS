import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_curve(values, labels, file_name):
    fig, ax1 = plt.subplots()
    color = next(ax1._get_lines.prop_cycler)['color']

    ax1.set_xlabel('Steps')
    ax1.plot(values[0], '--', color=color, label=labels[0])
    ax1.set_ylabel('Loss', color=color)
    ax1.tick_params('y', colors=color)

    ax1.set_xlabel('Steps')
    ax1.plot(values[1], '-', color=color, label=labels[1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=6)

    ax3 = ax1.twinx()
    i = 0
    for y_arr, label in zip(values[2:], labels[2:]):
        if i % 2 == 0:
            color = next(ax1._get_lines.prop_cycler)['color']
            ax3.plot(y_arr, '--', color=color, label=label)
        else:
            ax3.plot(y_arr, '-', color=color, label=label)

        i += 1

    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 0.2), loc=6)
    plt.savefig(file_name, bbox_inches='tight')


def plot_augmented_train():
    df_train = pd.read_csv('train.csv')
    df_train_augmented = pd.read_csv('train-augmented.csv')
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    for label in labels:
        df_train[label] = df_train['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
        df_train_augmented[label] = df_train_augmented['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


