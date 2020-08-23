import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# tags
dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','adxHistoricalDataset', 'generateHistoricalDatasets', 'experimentFolder', 'multivariateDatasets'))
REQUIRE_PLOTTING_SEPARATE_DATASETS = True
REQUIRE_PLOTTING_ALL_DATASETS_SIDE_BY_SIDE = {'Require':False, 'Normalize':False}

def plot_each_dataset_separately(list_of_files):
    for file in list_of_files:
        df = pd.read_csv(dir + '/' + file)
        df = df.fillna(0)
        # plt.figure(figsize=(100, 10))
        # plot as subplots - a subplot should consist of throughput, latency, error count etc. These subplots would be used to
        # manually label historical data
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        ax1.plot(df["timestamp"], df["throughput"], label="Throughput", linewidth=1)
        ax2.plot(df["timestamp"], df["avg_response_time"], label="Latency", linewidth=1)
        ax3.plot(df["timestamp"], df["http_error_count"], label="http_err_cnt", linewidth=1)
        ax4.plot(df["timestamp"], df["ballerina_error_count"], label="bal_err_cnt", linewidth=1)
        ax1.set_ylabel('Throughput')
        ax2.set_ylabel('Latency')
        ax3.set_ylabel('http_err_cnt')
        ax4.set_ylabel('bal_err_cnt')
        plt.savefig("visualisationPlots/"+file.strip(".csv")+".png")

def add_separator_column(df):
    df["separator_column"] = [0]*len(df)
    df.iloc[-1, df.columns.get_loc('separator_column')] = 1
    return df

def normalize_dataset(df):
    # Normalize the value column.
    max_val = np.percentile(df["throughput"], [99])[0]
    df["throughput"] = df["throughput"] / max_val
    return df

def plot_datasets_side_by_side(list_of_files):
    data_sets = [pd.read_csv(dir + '/' + file) for file in list_of_files]
    separated_datasets = [add_separator_column(df) for df in data_sets]
    if REQUIRE_PLOTTING_ALL_DATASETS_SIDE_BY_SIDE['Normalize']:
        separated_datasets = [normalize_dataset(df) for df in separated_datasets]
    data_set = pd.concat(separated_datasets)
    data_set = data_set.fillna(0)
    data_set = data_set.drop(["timestamp"], axis=1)
    data_set["Index"] = range(data_set.shape[0])
    data_set["is_separator"] = [
        data_set["throughput"].values[i] if data_set["separator_column"].values[i] == 1 else 0 for i in
        range(data_set.shape[0])]
    plt.figure(figsize=(100, 10))
    ax = sns.lineplot(x="Index", y="throughput", label="Throughput", data=data_set)
    ax = sns.scatterplot(x="Index", y="is_separator", data=data_set, ax=ax, label="Separation points", color='k', s=100)
    ax.legend()
    if REQUIRE_PLOTTING_ALL_DATASETS_SIDE_BY_SIDE['Normalize']:
        plt.savefig("visualisationPlots/full_dataset_normalized.png")
    else:
        plt.savefig("visualisationPlots/full_dataset_original.png")

def main():
    # Read each dataset in experimentFolder/processedMetrics from csv format to dataframe format
    list_of_files = [filename for filename in os.listdir(dir) if filename.endswith(".csv")]
    # This is the obsID/version of a long running test. This was removed temporary from plotting.
    if 'obsid_50eea11d-a588-4a3c-8acf-54cfacea8562|v-fb6901f0-4334-4cfe-ad08-fe6039260dd2.csv' in list_of_files:
        list_of_files.remove('obsid_50eea11d-a588-4a3c-8acf-54cfacea8562|v-fb6901f0-4334-4cfe-ad08-fe6039260dd2.csv')
    print(list_of_files)
    if REQUIRE_PLOTTING_SEPARATE_DATASETS:
        plot_each_dataset_separately(list_of_files)
    if REQUIRE_PLOTTING_ALL_DATASETS_SIDE_BY_SIDE['Require']:
        plot_datasets_side_by_side(list_of_files)

main()
