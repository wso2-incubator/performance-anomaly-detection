import numpy as np

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from plot_templates import *

from os import listdir
from os.path import isfile, join


def result_score(actual, predicted):
    if actual == predicted:
        return 5 if actual == 1 else 0
    elif actual == 0:
        return -3
    else:
        return -6


def plot_validation_sidebyside():
    df = pd.read_csv("results_prob.csv")
    df["Index"] = range(df.shape[0])
    actual = df["actual"].values
    predicted = df["adj_predicted"].values

    df["score"] = [result_score(actual[i], predicted[i]) for i in range(df.shape[0])]
    plt.figure(figsize=(100, 10))
    ax = sns.lineplot(x="Index", y="value", data=df)
    ax = sns.scatterplot(x="Index", y="score", data=df, ax=ax, color='r')
    ax.axhline(-3, ls='--', label="FP")
    ax.axhline(-6, ls='--', label="FN")
    plt.savefig("validation.png")

plot_validation_sidebyside()