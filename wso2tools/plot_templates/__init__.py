import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_waves(data_list, names, file_name, labels=None):
    graph_count = len(data_list)
    fig = plt.figure(figsize=(9, graph_count))
    for i, d in enumerate(data_list):
        ax = plt.subplot(graph_count, 1, i+1)
        ax.set_title(names[i])
        plt.plot(d)
        if i == 0 and labels is not None:
            labels = labels*np.mean(d)
            ax.plot(labels)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


def plot_xy(data_list, file_name):
    graph_count = len(data_list)
    fig = plt.figure(figsize=(5, graph_count))

    for i, (x,y,name) in enumerate(data_list):
        ax = plt.subplot(graph_count, 1, i+1)
        ax.set_title(name)
        plt.plot(x,y)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)

def plot_insame(data_list, file_name):
    graph_count = len(data_list)
    fig = plt.figure(figsize=(5, graph_count))

    for i, (x1,x2,name) in enumerate(data_list):
        ax = plt.subplot(graph_count, 1, i+1)
        ax.set_title(name)
        plt.plot(x1, label="1")
        plt.plot(x2, label="2")
        plt.legend()

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
