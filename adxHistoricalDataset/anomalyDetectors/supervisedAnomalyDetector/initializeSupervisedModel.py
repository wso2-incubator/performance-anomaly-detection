# Train the supervised anomaly detector using yws5 dataset and store in adxHistoricalDataset/anomalyDetectors/anomalyDetectorModels directory.
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import collections
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import *
import lightgbm
import pickle
import matplotlib.pyplot as plt

test_fraction = 0.3
value_feature_name_training = "value"

class Evaluation:
    def __init__(self, y_test, predict_test):
        self.accuracy = metrics.accuracy_score(y_test, predict_test)
        self.precision = metrics.precision_score(y_test, predict_test)
        self.recall = metrics.recall_score(y_test, predict_test)
        self.auc = metrics.roc_auc_score(y_test, predict_test)
        self.f1_score = metrics.f1_score(y_test, predict_test)
        self.cm = confusion_matrix(y_test, predict_test)

    def print(self):
        print("Accuracy\tPrecision\tRecall\tAUC\tF1")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (self.accuracy, self.precision, self.recall, self.auc, self.f1_score))

        print("Confusion Matrix")
        print(self.cm)

def adjust_predictions_for_neighbourhood(y_test, predict_test, slack=5):
    '''
    It is OK to forecast close to the anomaly as beginning of the anomaly is often not clear. This code check the neighbourhood
    and update the prediction. This help us get a better accuracy number.
    :param y_test:
    :param predict_test:
    :param slack: window to look at
    :return:
    '''
    y_test = y_test.values
    length = len(y_test)
    adjusted_forecasts = np.copy(predict_test)
    for i in range(length):
        if y_test[i] == predict_test[i]:
            adjusted_forecasts[i] = predict_test[i]
        elif predict_test[i] == 1: #FP
            if np.sum(y_test[i-slack:i+slack]) > 0:
                #print(y_test[i - slack:i + slack], "=", np.sum(y_test[i - slack:i + slack]))
                adjusted_forecasts[i] = 0 #there is anomaly within 20 in actual, so 1 OK
        elif predict_test[i] == 0:  # FN
            if np.sum(predict_test[i-slack:i+slack]) > 0:
                #print(predict_test[i - slack:i + slack], "=", np.sum(predict_test[i - slack:i + slack]))
                adjusted_forecasts[i] = 1 #there is anomaly within 20 in predicted, so OK
    return adjusted_forecasts

def plot_precision_recall(y_test, predict_test):
    '''
    Plot the precision recall tradeoff graph for give data to help find best cutoff
    :param y_test:
    :param predict_test:
    :return:
    '''
    precision, recall, thresholds = precision_recall_curve(y_test, predict_test)

    size = np.min([len(precision), len(recall), len(thresholds)])
    pr_tradeoff = pd.DataFrame()
    pr_tradeoff["precision"] = precision[:size]
    pr_tradeoff["recall"] = recall[:size]
    pr_tradeoff["thresholds"] = thresholds[:size]
    pr_tradeoff.to_csv("pr_tradeoff.csv")

    plt.plot(recall,precision, label="Precision")
    plt.ylabel("precision")
    plt.xlabel("recall")
    ax = plt.gca()
    plt.savefig("precision_recall.png")

class UnivariateAnomalyDetector:
    '''
    This is the main detector, can save and load to file system
    '''
    def __init__(self, filename=None):
        if filename is not None:
            self.clf = pickle.load(open(filename, 'rb'))
        else:
            self.clf = None

    #for now we need features to be generated before calling this
    def train(self, X_train, X_test, y_train, y_test, debug=False):
        params = {"scale_pos_weight": 0.05}
        params['learning_rate'] = 0.001
        d_train = lightgbm.Dataset(X_train, label=y_train)
        d_test = lightgbm.Dataset(X_test, label=y_test)

        params['learning_rate'] = 0.003
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 10
        params['min_data'] = 50
        params['max_depth'] = 10
        params['verbosity'] = 2
        print("using params", params)
        self.clf = lightgbm.train(params=params, train_set=d_train, num_boost_round=10000, valid_sets=d_test,
                        verbose_eval=True)
        predict_train = self.clf.predict(X_train)


        predict_test = self.clf.predict(X_test)
        importance_df = pd.DataFrame(sorted(zip(self.clf.feature_importance(), X_train.columns)),
                                     columns=['Value', 'Feature'])
        print(importance_df.sort_values(by=['Value'], ascending=False))
        plot_precision_recall(y_test, predict_test)

        if debug:
            X_test = X_test.copy()
            X_test["actual"] = y_test
            X_test["predicted"] = predict_test
            X_test.to_csv("results_prob.csv")

        predict_train, predict_test = [ 1 if v > 0.75 else 0 for v in predict_train], [ 1 if v > 0.75 else 0 for v in predict_test]
        eval = Evaluation(y_test, predict_test)
        eval.print()
        adjust_predictions = adjust_predictions_for_neighbourhood(y_test, predict_test)
        adj_eval = Evaluation(y_test, adjust_predictions)
        print("Adjusted Evaluations")
        adj_eval.print()
        return predict_train, predict_test

    def save_model(self, filename):
        pickle.dump(self.clf, open(filename, 'wb'))

    def predict(self, X_test):
        predict_test =  self.clf.predict(X_test)
        return  [1 if v > 0.75 else 0 for v in predict_test]

def under_sample_with_SMOTE(X, y):
    '''
    Undersample the date with SMOTE algorithm
    :param X:
    :param y: labels
    :return:
    '''
    counter = collections.Counter(y)
    print(counter)
    # define pipeline
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)
    # summarize the new class distribution
    counter = collections.Counter(y)
    print(counter)
    return X, y

def add_base_stat_features(s_train, prefix=""):
    window_list = [5, 25, 50]
    for i, w in enumerate(window_list):
        s_train[prefix + "value.w" + str(w) + ".mean"] = s_train["value"].rolling(w).mean().fillna(s_train["value"] if i ==0 else s_train[prefix + "value.w" + str(window_list[i-1]) + ".mean"])
        s_train[prefix + "value.w" + str(w) + ".std"] = s_train["value"].rolling(w).std().fillna(0 if i ==0 else s_train[prefix + "value.w" + str(window_list[i-1]) + ".std"])
        s_train[prefix + "value.w" + str(w) + ".kurt"] = s_train["value"].rolling(w).kurt().fillna(0 if i ==0 else s_train[prefix + "value.w" + str(window_list[i-1]) + ".kurt"])
        s_train[prefix + "value.w" + str(w) + ".zscore"] = s_train["value"].rolling(w).apply(lambda w: scipy.stats.zscore(w)[-1], raw=True)\
            .fillna(0 if i == 0 else s_train[prefix + "value.w" + str(window_list[i - 1]) + ".zscore"])
        s_train[prefix + "value.w" + str(w) + ".entropy"] = s_train["value"].rolling(w).apply(calculate_entropy, raw=True)\
            .fillna(0 if i == 0 else s_train[prefix + "value.w" + str(window_list[i - 1]) + ".entropy"])
    return s_train

def calculate_entropy(w):
    entropy = scipy.stats.entropy(w)
    return entropy if np.isfinite(entropy) else 10000

def calculate_ratio(s1, s2):
    l = len(s1.values)
    return [s1.values[i]/s2.values[i] if s2.values[i] != 0 and not(np.isnan(s2.values[i])) else 1000 for i in range(l)]

def get_seasonal_component(w):
    return get_seasonal_window(w)[-1]

def get_seasonal_window(w):
    frequencies = np.fft.fft(w)
    # TODO I am taking largest real value, is that right
    freq_size = np.sqrt(frequencies.real * frequencies.real + frequencies.imag * frequencies.imag)

    sortedIndexs = np.argsort(freq_size)  # returns values in ascending order

    index2keep = -1
    n = len(sortedIndexs)
    for i in range(n):
        if sortedIndexs[n - 1 - i] >= n / 2:
            index2keep = sortedIndexs[n - 1 - i]
            break

    period_freq = np.zeros(n, dtype=complex)
    period_freq[index2keep] = frequencies[index2keep]
    recovered_signal = np.fft.ifft(period_freq)
    return recovered_signal

def create_timeseries_features(s_train):
    s_train = add_base_stat_features(s_train)
    s_train["value.ratio"] = calculate_ratio(s_train["value"], s_train["value"].rolling(10).mean())
    s_train["value.w50.std.ratio"] = calculate_ratio(s_train["value.w50.std"], s_train["value.w50.std"].rolling(10).mean())
    s_train["value.w50.kurt.ratio"] = calculate_ratio(s_train["value.w50.kurt"], s_train["value.w50.kurt"].rolling(10).mean())
    s_train["value.w50.entropy.ratio"] = calculate_ratio(s_train["value.w50.entropy"], s_train["value.w50.entropy"].rolling(10).mean())
    s_train["value.w50.zscore.ratio"] = calculate_ratio(s_train["value.w50.zscore"],
                                                         s_train["value.w50.zscore"].rolling(10).mean())
    s_train["value.w50.mean.ratio"] = calculate_ratio(s_train["value.w50.mean"],
                                                         s_train["value.w50.mean"].rolling(10).mean())
    s_train["org.value"] = s_train["value"]
    s_train["value.trend_removed"] = s_train["value"] - s_train["value.w50.mean"]
    seasonal_component = s_train["value.trend_removed"].rolling(50).apply(get_seasonal_component, raw=False)
    seasonal_component = seasonal_component.fillna(0)
    s_train["value.season_trend_removed"] = s_train["value.trend_removed"] - seasonal_component
    # Before generating features
    s_train["value"] = s_train["value.season_trend_removed"]
    s_train = add_base_stat_features(s_train, prefix="nom.")
    s_train["nom.value.w50.std.ratio"] = calculate_ratio(s_train["nom.value.w50.std"], s_train["nom.value.w50.std"].rolling(10).mean())
    s_train["nom.value.w50.kurt.ratio"] = calculate_ratio(s_train["nom.value.w50.kurt"], s_train["nom.value.w50.kurt"].rolling(10).mean())
    s_train["nom.value.w50.entropy.ratio"] = calculate_ratio(s_train["nom.value.w50.entropy"], s_train["nom.value.w50.entropy"].rolling(10).mean())
    s_train["value"] = s_train["org.value"]
    s_train = s_train.drop(["org.value"], axis=1)
    return s_train

def normalize_dataset(df):
    max_val = np.percentile(df["value"], [90])[0]
    df["value"] = df["value"] / max_val
    return df

def load_yahoo_dataset():
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'yahooBenchmarkDataset',
                                       'ydata-labeled-time-series-anomalies-v1_0', 'A1Benchmark'))
    file_index_list = [11, 19, 15, 9, 45, 56, 59, 64, 41, 16, 54, 39, 2, 44, 23, 18, 47, 31, 24, 49, 26, 3, 0, 60, 52,
                       40, 43, 61, 12, 17, 33, 20, 1, 38, 7, 50, 30, 32, 21, 48, 5, 35, 4, 14, 29, 22, 55, 8, 62, 6, 58,
                       36, 53, 34, 10, 63, 28, 51, 25, 42, 57, 46, 13, 37, 27]
    file_list = [dir + "/real_" + str(fi + 1) + ".csv" for fi in file_index_list]
    data_sets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_dataset(df) for df in data_sets]
    transformed_datasets = [create_timeseries_features(df) for df in normalized_datasets]
    data_set = pd.concat(transformed_datasets)
    data_set = data_set.fillna(0)
    X = pd.DataFrame({"value": data_set[value_feature_name_training]})
    y = data_set.pop("is_anomaly")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=123,
                                                        shuffle=False)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Load yws5 dataset for the purpose of training
    X_train, X_test, y_train, y_test = load_yahoo_dataset()
    X_train, y_train = under_sample_with_SMOTE(X_train, y_train)
    # Train the supervised model using yws5 dataset
    detector = UnivariateAnomalyDetector()
    detector.train(X_train, X_test, y_train, y_test)
    # Save the trained model
    destination_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                       'anomalyDetectors', 'anomalyDetectorModels'))
    file_name = "/supervised_model.sav"
    detector.save_model(destination_directory+file_name)