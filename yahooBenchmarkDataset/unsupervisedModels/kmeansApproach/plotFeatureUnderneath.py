# imports
from yahooBenchmarkDataset.unsupervisedModels.kmeansApproach import datasetClassifier
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy

# list of all features. Features could be categorised as general features and specific features
    # (which have to be performed specifically only to detect specific types of anomalies)
ALL_AVAILABLE_FEATURES = ['value.lag1','value.lag2','value.lag3','value.lag5','value.lag10','value.lag15','first_diff','second_diff','first_long_diff','second_long_diff',
                          'value.w5.mean','value.w25.mean','value.w50.mean','value.w5.std','value.w25.std','value.w50.std','value.w5.kurt','value.w25.kurt','value.w50.kurt',
                          'value.w5.zscore','value.w25.zscore','value.w50.zscore','value.w5.entropy','value.w25.entropy','value.w50.entropy',
                          'value.lag2_ratio','value.lag3_ratio','value.lag5_ratio','value.lag10_ratio','value.lag15_ratio',
                          'value.ratio','first_diff_ratio','second_diff_ratio','value.w25.std.ratio','value.w25.kurt.ratio','value.w25.entropy.ratio','value.w25.zscore.ratio','value.w25.mean.ratio']

# tags
# 0 - spikes only, 1 - level shifts only, 2 - dips only, 3 - mixed anomalies, 4 - other types of anomalies (e.g. violation/change of seasonality), 5 - no anomalies
CATEGORY_LABELS = {0:'spikes_only',1:'level_shifts_only',2:'dips_only',3:'mixed_anomaly',4:'other_anomaly',5:'no_anomaly',6:'all_datasets_combined'}
FEATURE_VISUALIZATION_CATEGORICAL_DATASETS = {'Required':True, 'Categories':[0,1,2]}
# further tags for each type of anomaly.
# Plot all features underneath each type of dataset. That way, we could eliminate some features visually.
SPIKES_DETECTION_FEATURES = ['first_diff','second_diff','first_long_diff','second_long_diff','value.w5.mean','value.w25.mean',
                             'value.w50.mean','value.w5.std','value.w25.std','value.w50.std','value.w5.kurt','value.w25.kurt',
                             'value.w50.kurt','value.w5.zscore','value.w25.zscore','value.w50.zscore','value.w5.entropy',
                             'value.w25.entropy','value.w50.entropy', 'value.lag2_ratio','value.lag3_ratio','value.lag5_ratio',
                             'value.lag10_ratio','value.lag15_ratio','value.ratio','first_diff_ratio','second_diff_ratio',
                             'value.w25.std.ratio','value.w25.kurt.ratio','value.w25.entropy.ratio','value.w25.zscore.ratio','value.w25.mean.ratio']
LEVEL_SHIFTS_DETECTION_FEATURES = ['first_diff','second_diff','first_long_diff','second_long_diff','value.w5.mean','value.w25.mean',
                             'value.w50.mean','value.w5.std','value.w25.std','value.w50.std','value.w5.kurt','value.w25.kurt',
                             'value.w50.kurt','value.w5.zscore','value.w25.zscore','value.w50.zscore','value.w5.entropy',
                             'value.w25.entropy','value.w50.entropy', 'value.lag2_ratio','value.lag3_ratio','value.lag5_ratio',
                             'value.lag10_ratio','value.lag15_ratio','value.ratio','first_diff_ratio','second_diff_ratio',
                             'value.w25.std.ratio','value.w25.kurt.ratio','value.w25.entropy.ratio','value.w25.zscore.ratio','value.w25.mean.ratio']
DIPS_DETECTION_FEATURES = ['first_diff','second_diff','first_long_diff','second_long_diff','value.w5.mean','value.w25.mean',
                             'value.w50.mean','value.w5.std','value.w25.std','value.w50.std','value.w5.kurt','value.w25.kurt',
                             'value.w50.kurt','value.w5.zscore','value.w25.zscore','value.w50.zscore','value.w5.entropy',
                             'value.w25.entropy','value.w50.entropy', 'value.lag2_ratio','value.lag3_ratio','value.lag5_ratio',
                             'value.lag10_ratio','value.lag15_ratio','value.ratio','first_diff_ratio','second_diff_ratio',
                             'value.w25.std.ratio','value.w25.kurt.ratio','value.w25.entropy.ratio','value.w25.zscore.ratio','value.w25.mean.ratio']

def add_separator_column(df):
    df["separator_column"] = [0]*len(df)
    df.iloc[-1, df.columns.get_loc('separator_column')] = 1
    return df

def normalize_dataset(df):
    # Normalize the value column.
    max_val = np.percentile(df["value"], [99])[0]
    df["value"] = df["value"] / max_val
    return df

def load_dataset(category):
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','ydata-labeled-time-series-anomalies-v1_0','A1Benchmark'))
    switcher = {
        0 : datasetClassifier.Classifier.spikes_only,
        1 : datasetClassifier.Classifier.level_shifts_only,
        2 : datasetClassifier.Classifier.dips_only,
        3 : datasetClassifier.Classifier.mixed_anomaly,
        4 : datasetClassifier.Classifier.other_anomaly,
        5 : datasetClassifier.Classifier.no_anomaly,
        6 : datasetClassifier.Classifier.all_datasets_combined
    }
    list_of_files = ["/real_"+str(f)+".csv" for f in switcher[category]]
    data_sets = [pd.read_csv(dir+i) for i in list_of_files]
    separated_datasets = [add_separator_column(df) for df in data_sets]
    transformed_datasets = [normalize_dataset(df) for df in separated_datasets]
    data_set = pd.concat(transformed_datasets)
    data_set = data_set.fillna(0)
    data_set = data_set.drop(["timestamp"], axis=1)
    return data_set

def calculate_entropy(w):
    entropy = scipy.stats.entropy(w)
    return entropy if np.isfinite(entropy) else 10000

def calculate_ratio(s1, s2):
    l = len(s1.values)
    return [s1.values[i]/s2.values[i] if s2.values[i] != 0 and not(np.isnan(s2.values[i])) else 1000 for i in range(l)]

def create_timeseries_features(df):
    # To create lag features
    for i in [1, 2, 3, 5, 10, 15]:
        df["value.lag" + str(i)] = df["value"].shift(i).fillna(0)
    # To calculate first_diff and second_diff
    df["first_diff"] = [abs(df["value"].iloc[i] - df["value"].iloc[i-1]) if i > 1 else 0 for i in range(len(df["value"]))]
    df["second_diff"] = [abs(df["first_diff"].iloc[i] - df["first_diff"].iloc[i - 1]) if i > 1 else 0 for i in range(len(df["first_diff"]))]
    # To calculate first_long_diff and second_long_diff
    df["first_long_diff"] = abs(df["value.lag5"] - df["value.lag1"])
    df["second_long_diff"] = abs((df["value.lag10"] - df["value.lag5"]) - (df["value.lag5"] - df["value.lag1"]))
    # Basic statistical features
    window_list = [5, 25, 50]
    for i, w in enumerate(window_list):
        df["value.w" + str(w) + ".mean"] = df["value"].rolling(w).mean().fillna(
            df["value"] if i == 0 else df["value.w" + str(window_list[i - 1]) + ".mean"])
        df["value.w" + str(w) + ".std"] = df["value"].rolling(w).std().fillna(
            0 if i == 0 else df["value.w" + str(window_list[i - 1]) + ".std"])
        df["value.w" + str(w) + ".kurt"] = df["value"].rolling(w).kurt().fillna(
            0 if i == 0 else df["value.w" + str(window_list[i - 1]) + ".kurt"])
        df["value.w" + str(w) + ".zscore"] = df["value"].rolling(w).apply(
            lambda w: scipy.stats.zscore(w)[-1], raw=True) \
            .fillna(0 if i == 0 else df["value.w" + str(window_list[i - 1]) + ".zscore"])
        df["value.w" + str(w) + ".entropy"] = df["value"].rolling(w).apply(calculate_entropy,raw=True) \
            .fillna(0 if i == 0 else df["value.w" + str(window_list[i - 1]) + ".entropy"])
    # Features using ratio
    for i in [2, 3, 5, 10, 15]:
        df["value.lag" + str(i) + "_ratio"] = calculate_ratio(df["value.lag" + str(i)], df["value.lag1"])
    df["value.ratio"] = calculate_ratio(df["value"], df["value"].rolling(10).mean())
    df["first_diff_ratio"] = [df["value"].iloc[i] / df["value"].iloc[i - 1] if i > 1 and df["value"].iloc[i - 1] > 0 else 1000 for i in range(len(df["value"]))]
    df["second_diff_ratio"] = [df["first_diff_ratio"].iloc[i] / df["first_diff_ratio"].iloc[i - 1] if i > 1 and df["first_diff_ratio"].iloc[i - 1] > 0 else 1000 for i in
                               range(len(df["first_diff_ratio"]))]
    df["value.w25.std.ratio"] = calculate_ratio(df["value.w25.std"],
                                                     df["value.w25.std"].rolling(10).mean())
    df["value.w25.kurt.ratio"] = calculate_ratio(df["value.w25.kurt"],
                                                      df["value.w25.kurt"].rolling(10).mean())
    df["value.w25.entropy.ratio"] = calculate_ratio(df["value.w25.entropy"],
                                                         df["value.w25.entropy"].rolling(10).mean())
    df["value.w25.zscore.ratio"] = calculate_ratio(df["value.w25.zscore"],
                                                        df["value.w25.zscore"].rolling(10).mean())
    df["value.w25.mean.ratio"] = calculate_ratio(df["value.w25.mean"],
                                                      df["value.w25.mean"].rolling(10).mean())
    return df

def plot_selected_feature_underneath(required_categories):
    for category in required_categories:
        df = load_dataset(category)
        df = create_timeseries_features(df)
        df["Index"] = range(df.shape[0])
        df["is_anomaly_labels"] = [df["value"].values[i] if df["is_anomaly"].values[i] == 1 else -0.5 for i in
                                   range(df.shape[0])]
        switcher = {
            0: SPIKES_DETECTION_FEATURES,
            1: LEVEL_SHIFTS_DETECTION_FEATURES,
            2: DIPS_DETECTION_FEATURES, # Add 3,4,5 as required
        }
        detination_folder = {
            0: 'spikes_only',
            1: 'level_shifts_only',
            2: 'dips_only'
        }
        for feature in switcher[category]:
            if not os.path.exists("visualizationFigures/featurePlots/" + str(detination_folder[category]) + "/feature_plot" + str(feature) + ".png"):
                fig, (ax1, ax2) = plt.subplots(2)
                fig.suptitle('Vertically stacked subplots of value and feature '+str(feature))
                ax1.plot(df["Index"], df["value"], linewidth=1)
                # Plot the anomalies on top of the linechart
                for x,y in zip(df["Index"], df["is_anomaly_labels"]):
                    ax1.scatter(x,y,marker='o',color='r')
                ax2.plot(df["Index"], df[feature], linewidth=1)
                plt.savefig("visualizationFigures/featurePlots/"+str(detination_folder[category])+"/feature_plot" + str(feature) + ".png")

def main():
    if FEATURE_VISUALIZATION_CATEGORICAL_DATASETS['Required']:
        plot_selected_feature_underneath(FEATURE_VISUALIZATION_CATEGORICAL_DATASETS['Categories'])

main()