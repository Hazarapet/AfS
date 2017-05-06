import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_curve(values, title, file_name, x_axis='Step', y_axis='Loss'):
    plt.figure(title)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel(y_axis, fontsize=12)
    plt.plot(values, '-b', label=title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
    plt.savefig(file_name, bbox_inches='tight')