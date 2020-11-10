# Python program to read A1Benchmark from Yahoo dataset, create time-series features out of it and save it as a csv
# Imports
import os
import pandas as pd
import numpy as np
import scipy
from scipy import stats

# Global variables
TIMESTAMP_COLUMN_NAME = "timestamp"
ANOMALY_LABEL_COLUMN_NAME = "is_anomaly"
UNIVARIATE_COLUMN_NAME = "value"
FEATURE_COLUMN_PREFIX = "value.w"

PERCENTILE_VALUE = 90
SUPERVISED_WINDOW_LIST = [5, 25, 50]
DEFAULT_RATIO_VALUE = 1000
MAX_ENTROPY_VALUE = 10000

# Functions
def load_training_dataset():
    """
    Load the training dataset which will train the SupervisedAnomalyDetector.
    :return: X_train, X_test, y_train, y_test
    """
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','ydata-labeled-time-series-anomalies-v1_0',
                                       'A1Benchmark'))
    file_index_list = [11, 19, 15, 9, 45, 56, 59, 64, 41, 16, 54, 39, 2, 44, 23, 18, 47, 31, 24, 49, 26, 3, 0, 60,
                       52, 40, 43, 61, 12, 17, 33, 20, 1, 38, 7, 50, 30, 32, 21, 48, 5, 35, 4, 14, 29, 22, 55, 8, 62,
                       6, 58, 36, 53, 34, 10, 63, 28, 51, 25, 42, 57, 46, 13, 37, 27]
    file_list = [dir + "/real_" + str(fi + 1) + ".csv" for fi in file_index_list]
    data_sets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_dataset(df) for df in data_sets]
    transformed_datasets = [create_timeseries_features(df) for df in normalized_datasets]
    data_set = pd.concat(transformed_datasets)
    data_set = data_set.fillna(0)
    X = data_set.drop([TIMESTAMP_COLUMN_NAME], axis=1)
    return X

def create_timeseries_features(df):
    """
    Create timeseries features for the loaded dataset, which are required to train the SupervisedAnomalyDetector model.
    :param df: Input dataframe
    :return: Dataframe with time-series features calculated
    """
    df = add_base_stat_features(df)
    df[UNIVARIATE_COLUMN_NAME+".ratio"] = calculate_ratio(df[UNIVARIATE_COLUMN_NAME], df[UNIVARIATE_COLUMN_NAME].rolling(10).mean())
    df[FEATURE_COLUMN_PREFIX+"50.std.ratio"] = calculate_ratio(df[FEATURE_COLUMN_PREFIX+"50.std"],
                                                     df[FEATURE_COLUMN_PREFIX+"50.std"].rolling(10).mean())
    df[FEATURE_COLUMN_PREFIX+"50.kurt.ratio"] = calculate_ratio(df[FEATURE_COLUMN_PREFIX+"50.kurt"],
                                                      df[FEATURE_COLUMN_PREFIX+"50.kurt"].rolling(10).mean())
    df[FEATURE_COLUMN_PREFIX+"50.entropy.ratio"] = calculate_ratio(df[FEATURE_COLUMN_PREFIX+"50.entropy"],
                                                         df[FEATURE_COLUMN_PREFIX+"50.entropy"].rolling(10).mean())
    df[FEATURE_COLUMN_PREFIX+"50.zscore.ratio"] = calculate_ratio(df[FEATURE_COLUMN_PREFIX+"50.zscore"],
                                                        df[FEATURE_COLUMN_PREFIX+"50.zscore"].rolling(10).mean())
    df[FEATURE_COLUMN_PREFIX+"50.mean.ratio"] = calculate_ratio(df[FEATURE_COLUMN_PREFIX+"50.mean"],
                                                      df[FEATURE_COLUMN_PREFIX+"50.mean"].rolling(10).mean())
    # "org.value" is an intermediate column used inside the function.
    df["org.value"] = df[UNIVARIATE_COLUMN_NAME]
    df[UNIVARIATE_COLUMN_NAME+".trend_removed"] = df[UNIVARIATE_COLUMN_NAME] - df[FEATURE_COLUMN_PREFIX+"50.mean"]
    seasonal_component = df[UNIVARIATE_COLUMN_NAME+".trend_removed"].rolling(50).apply(get_seasonal_component, raw=False)
    seasonal_component = seasonal_component.fillna(0)
    df[UNIVARIATE_COLUMN_NAME+".season_trend_removed"] = df[UNIVARIATE_COLUMN_NAME+".trend_removed"] - seasonal_component
    # Before generating features
    df[UNIVARIATE_COLUMN_NAME] = df[UNIVARIATE_COLUMN_NAME+".season_trend_removed"]
    df = add_base_stat_features(df, prefix="nom.")
    df["nom."+FEATURE_COLUMN_PREFIX+"50.std.ratio"] = calculate_ratio(df["nom."+FEATURE_COLUMN_PREFIX+"50.std"],
                                                         df["nom."+FEATURE_COLUMN_PREFIX+"50.std"].rolling(10).mean())
    df["nom."+FEATURE_COLUMN_PREFIX+"50.kurt.ratio"] = calculate_ratio(df["nom."+FEATURE_COLUMN_PREFIX+"50.kurt"],
                                                          df["nom."+FEATURE_COLUMN_PREFIX+"50.kurt"].rolling(10).mean())
    df["nom."+FEATURE_COLUMN_PREFIX+"50.entropy.ratio"] = calculate_ratio(df["nom."+FEATURE_COLUMN_PREFIX+"50.entropy"],
                                                             df["nom."+FEATURE_COLUMN_PREFIX+"50.entropy"].rolling(10).mean())
    df[UNIVARIATE_COLUMN_NAME] = df["org.value"]
    df = df.drop(["org.value"], axis=1)

    return df

def normalize_dataset(df, max_val=None):
    """
    Normalise a dataset by dividing the univariate value column by the 90th percentile of the dataset.
    :param df: Input dataframe
    :param max_val: The 90th percentile value that will be passed along with function invokation. If max_val is None
        this function will calculate the 90th percentile value of the input dataframe.
    :return: Normalised dataset after dividing the univariate value column by the 90th percentile.
    """
    if max_val == None:
        max_val = np.percentile(df[UNIVARIATE_COLUMN_NAME], [PERCENTILE_VALUE])[0]
    df[UNIVARIATE_COLUMN_NAME] = df[UNIVARIATE_COLUMN_NAME] / max_val
    return df

# Special functions only for SupervisedAnomalyDetector
def add_base_stat_features(df, prefix=""):
    """
    Create basic statistical features such as mean, std, kurtosis, zscore and entropy for a given dataframe for the
    window values specified in SUPERVISED_WINDOW_LIST.
    :param df: Input dataframe
    :param prefix:
    :return: Dataframe with basic statistical features calculated
    """
    # i stands for index and w stands for window in window_list
    for i, w in enumerate(SUPERVISED_WINDOW_LIST):
        df[prefix + FEATURE_COLUMN_PREFIX + str(w) + ".mean"] = df[UNIVARIATE_COLUMN_NAME].rolling(w).mean().fillna(
            df[UNIVARIATE_COLUMN_NAME] if i == 0 else df[prefix + FEATURE_COLUMN_PREFIX + str(SUPERVISED_WINDOW_LIST[i - 1]) + ".mean"])
        df[prefix + FEATURE_COLUMN_PREFIX + str(w) + ".std"] = df[UNIVARIATE_COLUMN_NAME].rolling(w).std().fillna(
            0 if i == 0 else df[prefix + FEATURE_COLUMN_PREFIX + str(SUPERVISED_WINDOW_LIST[i - 1]) + ".std"])
        df[prefix + FEATURE_COLUMN_PREFIX + str(w) + ".kurt"] = df[UNIVARIATE_COLUMN_NAME].rolling(w).kurt().fillna(
            0 if i == 0 else df[prefix + FEATURE_COLUMN_PREFIX + str(SUPERVISED_WINDOW_LIST[i - 1]) + ".kurt"])
        df[prefix + FEATURE_COLUMN_PREFIX + str(w) + ".zscore"] = df[UNIVARIATE_COLUMN_NAME].rolling(w).apply(
            lambda w: scipy.stats.zscore(w)[-1], raw=True) \
            .fillna(0 if i == 0 else df[prefix + FEATURE_COLUMN_PREFIX + str(SUPERVISED_WINDOW_LIST[i - 1]) + ".zscore"])
        df[prefix + FEATURE_COLUMN_PREFIX + str(w) + ".entropy"] = df[UNIVARIATE_COLUMN_NAME].rolling(w).apply(calculate_entropy,
                                                                                              raw=True) \
            .fillna(0 if i == 0 else df[prefix + FEATURE_COLUMN_PREFIX + str(SUPERVISED_WINDOW_LIST[i - 1]) + ".entropy"])
    return df

def calculate_entropy(window):
    """
    Calculate entropy for a given window.
    :param window:
    :return: Calculated entropy for the given window.
    """
    entropy = scipy.stats.entropy(window)
    return entropy if np.isfinite(entropy) else MAX_ENTROPY_VALUE

def calculate_ratio(series_1, series_2):
    """
    Calculate and return the ratio between series_1 and series_2.
    :param series_1: a Pandas series containing floating point values
    :param series_2: a Pandas series containing floating point values
    :return: the ratio between series_1 and series_2
    """
    l = len(series_1.values)
    return [series_1.values[i] / series_2.values[i] if series_2.values[i] != 0 and not (np.isnan(series_2.values[i])) else DEFAULT_RATIO_VALUE for i in
            range(l)]

def get_seasonal_component(window):
    """
    Return the last value obtained by invoking the get_seasonal_window(window) function.
    :param window:
    :return: the last value obtained by invoking the get_seasonal_window(window) function
    """
    return get_seasonal_window(window)[-1]

def get_seasonal_window(window):
    """
    Obtain the seasonal window for a given window
    :param window: window in the form of input array, (according to the docs, this value could be complex)
    :return: recovered_signal
    """
    frequencies = np.fft.fft(window)
    freq_size = np.sqrt(frequencies.real * frequencies.real + frequencies.imag * frequencies.imag)

    sortedIndices = np.argsort(freq_size)  # returns values in ascending order

    index_to_keep = -1
    n = len(sortedIndices)
    for i in range(n):
        if sortedIndices[n - 1 - i] >= n / 2:
            index_to_keep = sortedIndices[n - 1 - i]
            break

    period_freq = np.zeros(n, dtype=complex)
    period_freq[index_to_keep] = frequencies[index_to_keep]
    recovered_signal = np.fft.ifft(period_freq)
    return recovered_signal

if __name__ == '__main__':
    df = load_training_dataset()
    # Save the newly created dataset as a csv
    df.to_csv("dataset/yws5_preprocessed.csv")