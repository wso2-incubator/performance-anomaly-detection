# This code reads the A1Benchmark of the yws5 and merge all csvs into the same time-series
# Steps to be followed
# 1) Normalize/Standardize the dataset by dividing each time-series by the 90th percentiles
# 2) Merge all datasets into one time-series dataset and reindex
# Final format of dataset would be - timestamp, value, is_anomaly
# This pre-processing is being done with the aim of evaluating the performance of DevNet with an LSTM/GRU layer.

import os
import pandas as pd
import numpy as np

# Global variables
TIMESTAMP_COLUMN_NAME = "timestamp"
UNIVARIATE_COLUMN_NAME = "value"
PERCENTILE_VALUE = 90

def normalize_dataset(df, max_val=None):
    if max_val == None:
        max_val = np.percentile(df[UNIVARIATE_COLUMN_NAME], [PERCENTILE_VALUE])[0]
    df[UNIVARIATE_COLUMN_NAME] = df[UNIVARIATE_COLUMN_NAME] / max_val
    return df


def load_dataset():
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ydata-labeled-time-series-anomalies-v1_0',
                                       'A1Benchmark'))
    file_index_list = [11, 19, 15, 9, 45, 56, 59, 64, 41, 16, 54, 39, 2, 44, 23, 18, 47, 31, 24, 49, 26, 3, 0, 60,
                       52, 40, 43, 61, 12, 17, 33, 20, 1, 38, 7, 50, 30, 32, 21, 48, 5, 35, 4, 14, 29, 22, 55, 8, 62,
                       6, 58, 36, 53, 34, 10, 63, 28, 51, 25, 42, 57, 46, 13, 37, 27]
    file_list = [dir + "/real_" + str(fi + 1) + ".csv" for fi in file_index_list]
    data_sets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_dataset(df) for df in data_sets]
    data_set = pd.concat(normalized_datasets)
    data_set = data_set.fillna(0)
    X = data_set.drop([TIMESTAMP_COLUMN_NAME], axis=1)
    return X

if __name__ == '__main__':
    df = load_dataset()
    df = df.reset_index(drop=True)
    # Save the newly created dataset as a csv
    df.to_csv("dataset/yws5_for_lstm.csv")