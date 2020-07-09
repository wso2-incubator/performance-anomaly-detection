# Implementation of the combined solution based only on features.
# imports
from yahooBenchmarkDataset.unsupervisedModels.kmeansApproach import datasetClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter1d
import os
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

# Global features set
# list of all features. Features could be categorised as general features and specific features
    # (which have to be performed specifically only to detect specific types of anomalies)
ALL_AVAILABLE_FEATURES = ['value.lag1','value.lag2','value.lag3','value.lag5','value.lag10','value.lag15','first_diff','second_diff','first_long_diff','second_long_diff',
                          'value.w5.mean','value.w25.mean','value.w50.mean','value.w5.std','value.w25.std','value.w50.std','value.w5.kurt','value.w25.kurt','value.w50.kurt',
                          'value.w5.zscore','value.w25.zscore','value.w50.zscore','value.w5.entropy','value.w25.entropy','value.w50.entropy',
                          'value.lag2_ratio','value.lag3_ratio','value.lag5_ratio','value.lag10_ratio','value.lag15_ratio',
                          'value.ratio','first_diff_ratio','second_diff_ratio','value.w25.std.ratio','value.w25.kurt.ratio','value.w25.entropy.ratio','value.w25.zscore.ratio','value.w25.mean.ratio']
ATTACK_BOUNDARY_PERCENTILE = 99
RELAX_EVALUATION_CONDITION = True

# tags
# 0 - spikes only, 1 - level shifts only, 2 - dips only, 3 - mixed anomalies, 4 - other types of anomalies (e.g. violation/change of seasonality), 5 - no anomalies
CATEGORY_LABELS = {0:'spikes_only',1:'level_shifts_only',2:'dips_only',3:'mixed_anomaly',4:'other_anomaly',5:'no_anomaly',6:'all_datasets_combined'}
COMBINED_ANOMALY_DETECTION_DATASETS = {'Required':True, 'Categories':[3,6]}
switcher = {
        0 : datasetClassifier.Classifier.spikes_only,
        1 : datasetClassifier.Classifier.level_shifts_only,
        2 : datasetClassifier.Classifier.dips_only,
        3 : datasetClassifier.Classifier.mixed_anomaly,
        4 : datasetClassifier.Classifier.other_anomaly,
        5 : datasetClassifier.Classifier.no_anomaly,
        6 : datasetClassifier.Classifier.all_datasets_combined
    }
# further tags for each type of anomaly
SPIKES_DETECTION_FEATURES = ['value.w25.mean','value.w25.std','value.w25.zscore','value.w50.mean','value.w50.std','value.w50.zscore']
LEVEL_SHIFTS_DETECTION_FEATURES = ['value.w25.mean', 'value.w25.std', 'value.w50.mean', 'value.w50.std']
DIPS_DETECTION_FEATURES = ['value.w5.entropy', 'value.w25.entropy', 'value.w50.entropy']

spikes_requirements = {'feature_list':SPIKES_DETECTION_FEATURES, 'IS_SIGNAL_PROCESS':[False], 'OPTIMAL_K_VALUE':8, 'VALUE_IS_FEATURE':True, 'INCLUDE_RESIDUAL':False}
level_shifts_requirements = {'feature_list':LEVEL_SHIFTS_DETECTION_FEATURES, 'IS_SIGNAL_PROCESS':[True,3], 'OPTIMAL_K_VALUE':4, 'VALUE_IS_FEATURE':True, 'INCLUDE_RESIDUAL':False}
dips_requirements = {'feature_list':DIPS_DETECTION_FEATURES, 'IS_SIGNAL_PROCESS':[False], 'OPTIMAL_K_VALUE':4, 'VALUE_IS_FEATURE':False, 'INCLUDE_RESIDUAL':True}
requirements_list = {
        0 : spikes_requirements,
        1 : level_shifts_requirements,
        2 : dips_requirements,
    }

def add_separator_column(df):
    df["separator_column"] = [0]*len(df)
    df.iloc[-1, df.columns.get_loc('separator_column')] = 1
    return df

def normalize_dataset(df):
    # Normalize the value column. Generally dividing by 90th percentile is good.
    max_val = np.percentile(df["value"], [90])[0]
    df["value"] = df["value"] / max_val
    return df

def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]
    if np.abs(r) > 0.5 and lag>1:
        return lag

def remove_seasonality(df):
    # detect if there is any seasonality in df by using auto-correlation plots
    period = autocorr(df["value"])
    if period is not None:
        # perform season decompose and replace df["value"] with residual component
        seas_decompose = seasonal_decompose(df["value"], model='additive', period=period)
        df["value"] = seas_decompose.resid
        return df
    else:
        return df

def remove_noise(df):
    df["value"] = gaussian_filter1d(df["value"], 3)
    return df

def load_dataset(category):
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','ydata-labeled-time-series-anomalies-v1_0','A1Benchmark'))
    list_of_files = ["/real_"+str(f)+".csv" for f in switcher[category]]
    data_sets = [pd.read_csv(dir+i) for i in list_of_files]
    separated_datasets = [add_separator_column(df) for df in data_sets]
    normalized_datasets = [normalize_dataset(df) for df in separated_datasets]
    # Before creating timeseries features,FIRST remove any seasonality.
    # residual_datasets = [remove_seasonality(df) for df in normalized_datasets]
    # Before creating timeseries features,SECONDLY remove any noise.
    noise_free_datasets = [remove_noise(df) for df in normalized_datasets]
    transformed_datasets = [create_timeseries_features(df) for df in noise_free_datasets]
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

def calculate_cluster_boundaries(x_dist, labels, percentile_threshold):
    boundary_list = []
    for cluster in np.unique(labels):
        # indexes in the current cluster. Labels are grouped in to clusters.
        # e.g.: a=[1, 2, 3, 1, 2] -> corresponding index_list is [0 3][1 4][2]
        index_list = np.where(labels == cluster)[0]

        # distances to relevant cluster centroid
        distances = x_dist[index_list, cluster]
        boundary = np.percentile(distances, percentile_threshold)
        boundary_list.append(boundary)

    return boundary_list

def train_model(train_data, category):
    optimal_k = requirements_list[category]["OPTIMAL_K_VALUE"] #This value can be automatically decided by using Silhoutte method. But initially this could be decided by visually observing clusters
    km = KMeans(optimal_k, max_iter=800, algorithm='auto', random_state=50)
    km.fit(train_data)
    best_classifier = km

    # Calculating and saving the cluster boundaries for attack detection
    x_distances = best_classifier.transform(train_data) # Transform train_data to a cluster-distance space.

    # In the new space, each dimension is the distance to the cluster centers.
    labels = best_classifier.labels_

    # attack and warn boundaries for each cluster is
    # print("Attack cluster boundaries: ")
    attack_boundary_list = calculate_cluster_boundaries(x_distances, labels, ATTACK_BOUNDARY_PERCENTILE)
    # print(attack_boundary_list)

    return best_classifier, attack_boundary_list

# Prediction as to whether a data point is anomalous or not
def predict_data_point(data_point, classifier, attack_boundary_list):
    # data_point is a pandas data frame that contains features for the data point.

    predictions = classifier.predict(data_point)
    distances = classifier.transform(data_point)
    dis = distances[0][predictions[0]]
    attack_cluster_boundary = attack_boundary_list[predictions[0]]

    is_attack = False

    if dis > attack_cluster_boundary:
        is_attack = True

    return is_attack

# Evaluate against y_test
def print_evaluations(label_list, detection_list):
    accuracy = metrics.accuracy_score(label_list, detection_list)
    precision = metrics.precision_score(label_list, detection_list)
    recall = metrics.recall_score(label_list, detection_list)
    try:
        auc = metrics.roc_auc_score(label_list, detection_list)
    except:
        auc = 0

    f1_score = metrics.f1_score(label_list, detection_list)

    resultsdf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "F1"], data=[[accuracy, precision, recall, auc, f1_score]])
    print(resultsdf)

    print("Confusion Matrix")
    cm = confusion_matrix(label_list, detection_list)
    print(cm)

    return accuracy, precision, recall, auc, f1_score

def plot_for_evaluation(X_test, y_test, detection_list, category):
    X_test["is_anomaly"] = y_test
    X_test["Index"] = range(X_test.shape[0])
    X_test["detection_list"] = detection_list
    X_test["tp"] = [X_test["value"].values[i] if X_test["is_anomaly"].values[i] == 1 and X_test["detection_list"].values[i] == 1 else -0.5 for i in
                               range(X_test.shape[0])]
    X_test["fp"] = [X_test["value"].values[i] if X_test["is_anomaly"].values[i] == 0 and X_test["detection_list"].values[i] == 1 else -0.5 for i in
                    range(X_test.shape[0])]
    X_test["fn"] = [X_test["value"].values[i] if X_test["is_anomaly"].values[i] == 1 and X_test["detection_list"].values[i] == 0 else -0.5 for i in
                    range(X_test.shape[0])]
    plt.figure(figsize=(100, 10))
    ax = sns.lineplot(x="Index", y="value", data=X_test)
    ax = sns.scatterplot(x="Index", y="tp", data=X_test, ax=ax, label="True Positive", color='r', s=100)
    ax = sns.scatterplot(x="Index", y="fp", data=X_test, ax=ax, label="False Positive", color='g', s=100)
    ax = sns.scatterplot(x="Index", y="fn", data=X_test, ax=ax, label="False Negative", color='darkorange', s=100)
    ax.legend()
    detination_folder = {
        3: 'mixed_anomaly',
        6: 'all_datasets_combined'
    }
    plt.savefig("visualizationFigures/afterDetectionPlots/" + str(detination_folder[category]) + "/after_detection_plot.png")

def detect_specific_type_of_anomaly(df,category):
    # if requirements_list[category]['IS_SIGNAL_PROCESS'][0]:
    #     df["value"] = gaussian_filter1d(df["value"], requirements_list[category]['IS_SIGNAL_PROCESS'][1])
    # if requirements_list[category]['INCLUDE_RESIDUAL']:
    #     df["residual_feature"] = df["value"] - savgol_filter(df["value"], 601, 2)
    #     # print(df.columns[df.isna().any()].tolist())
    #     result = seasonal_decompose(df["value"], model='additive', period=2)
    #     df["season_dash"] = (result.resid).fillna(0)
    #     # print(df.columns[df.isna().any()].tolist())
    # drop unwanted features based on category
    drop_features_list = list(set(ALL_AVAILABLE_FEATURES) - set(requirements_list[category]['feature_list']))
    drop_features_list.append('separator_column')
    df = df.drop(drop_features_list, axis=1)
    # split dataset
    y = df.pop("is_anomaly")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=123,
                                                        shuffle=False)
    # Send only non-anomalous data to train.
    # Compare X_train with y_train and remove anomalous data.
    training_set_labels = list(y_train.values)
    anomaly_index_list = []
    for i in range(len(training_set_labels)):
        if training_set_labels[i] == 1:
            anomaly_index_list.append(i)
    X_train.reset_index(drop=True, inplace=True)
    X_train_normal_data = X_train.drop(X_train.index[anomaly_index_list])  # X_train_normal_data is the training set without anomalies
    if not requirements_list[category]['VALUE_IS_FEATURE']:
        X_train_normal_data = X_train_normal_data.drop(['value'], axis=1)
    # Send X_train_normal_data for training
    best_classifier, attack_boundary_list = train_model(X_train_normal_data, category)

    detection_list = []
    for i in range(len(X_test.index)):
        data_point = X_test.iloc[[i]]  # data_point is a dataframe
        if not requirements_list[category]['VALUE_IS_FEATURE']:
            data_point = data_point.drop(['value'], axis=1)
        is_attack = predict_data_point(data_point, best_classifier, attack_boundary_list)
        if is_attack:
            detection_list.append(1)
        else:
            detection_list.append(0)
    return (X_test,y_test,detection_list)

def relaxed_evaluation_condition(y_test, predict_test):
    slack = 5
    y_test = y_test.values
    length = len(y_test)
    adjusted_forecasts = np.copy(predict_test)
    for i in range(length):
        if y_test[i] == predict_test[i]:
            adjusted_forecasts[i] = predict_test[i]
        elif predict_test[i] == 1:  # FP
            if np.sum(y_test[i - slack:i + slack]) > 0:
                # print(y_test[i - slack:i + slack], "=", np.sum(y_test[i - slack:i + slack]))
                adjusted_forecasts[i] = 0  # there is anomaly within 20 in actual, so 1 OK
        elif predict_test[i] == 0:  # FN
            if np.sum(predict_test[i - slack:i + slack]) > 0:
                # print(predict_test[i - slack:i + slack], "=", np.sum(predict_test[i - slack:i + slack]))
                adjusted_forecasts[i] = 1  # there is anomaly within 20 in predicted, so OK
    return adjusted_forecasts


def main():
    if COMBINED_ANOMALY_DETECTION_DATASETS['Required']:
        for category in COMBINED_ANOMALY_DETECTION_DATASETS['Categories']:
            df = load_dataset(category)
            # df = create_timeseries_features(df)
            # detect spikes
            X_test_spikes,y_test_spikes,spikes_detection_list = detect_specific_type_of_anomaly(df,0)
            # detect level_shifts
            X_test_level_shifts,y_test_level_shifts,level_shift_detection_list = detect_specific_type_of_anomaly(df,1)
            # detect dips
            X_test_dips,y_test_dips,dips_detection_list = detect_specific_type_of_anomaly(df,2)
            # combine the detected anomalies
            final_detection_list=[]
            for i in range(len(spikes_detection_list)):
                if spikes_detection_list[i]==0 and level_shift_detection_list[i]==0 and dips_detection_list[i]==0:
                    final_detection_list.append(0)
                elif spikes_detection_list[i]==1 or level_shift_detection_list[i]==1 or dips_detection_list[i]==1:
                    final_detection_list.append(1)
            print(CATEGORY_LABELS[category])
            # Invoke a relaxed evaluation condition
            if RELAX_EVALUATION_CONDITION:
                adjusted_prediction_list=relaxed_evaluation_condition(y_test_spikes, final_detection_list)
                print_evaluations(y_test_spikes, adjusted_prediction_list)
                # Plot only if required
                plot_for_evaluation(X_test_spikes, y_test_spikes, adjusted_prediction_list, category)
            else:
                print_evaluations(y_test_spikes, final_detection_list)  # y_test_spikes is the label_list
                # After execution, plot the TP, FP and FN datapoints in the dataset
                plot_for_evaluation(X_test_spikes, y_test_spikes, final_detection_list, category)

main()
# mixed_anomaly
#    Accuracy  Precision    Recall       AUC        F1
# 0  0.962941   0.259887  0.282209  0.631072  0.270588
# Confusion Matrix
# [[6398  131]
#  [ 117   46]]
#
# all_datasets_combined
#    Accuracy  Precision    Recall       AUC        F1
# 0  0.974491    0.17623  0.511905  0.745264  0.262195
# Confusion Matrix
# [[27605   603]
#  [  123   129]]

# After adjusting
# mixed_anomaly
#    Accuracy  Precision    Recall       AUC       F1
# 0  0.973102   0.460465  0.607362  0.794798  0.52381
# Confusion Matrix
# [[6413  116]
#  [  64   99]]
# all_datasets_combined
#    Accuracy  Precision    Recall       AUC        F1
# 0  0.978707   0.226852  0.583333  0.782786  0.326667
# Confusion Matrix
# [[27707   501]
#  [  105   147]]