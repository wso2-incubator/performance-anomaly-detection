from wso2.time_series import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

import lightgbm as lgb
import matplotlib.pyplot as plt
import imblearn
import collections
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle

def plot_precision_recall(y_test, predict_test):
    precision, recall, thresholds = precision_recall_curve(y_test, predict_test)

    size = np.min([len(precision), len(recall), len(thresholds)])
    pr_tradeoff = pd.DataFrame()
    pr_tradeoff["precision"] = precision[:size]
    pr_tradeoff["recall"] = recall[:size]
    pr_tradeoff["thresholds"] = thresholds[:size]
    pr_tradeoff.to_csv("pr_tradeoff.csv")

    pr_tradeoff = pr_tradeoff[pr_tradeoff["precision"] > 0.75]
    pr_tradeoff = pr_tradeoff.sort_values(by=["precision"])
    top_record = pr_tradeoff.head(1)
    print("recall at 0.75")
    print(top_record)

    plt.plot(recall,precision, label="Precision")
    plt.ylabel("precision")
    plt.xlabel("recall")
    ax = plt.gca()
    #secaxy = ax.plot(recall[:len(thresholds)], thresholds, label="threshold")
    plt.savefig("precision_recall.png")

class Evaluation:
    def __init__(self, y_test, predict_test):
        self.acccuracy = metrics.accuracy_score(y_test, predict_test)
        self.preceission = metrics.precision_score(y_test, predict_test)
        self.recall = metrics.recall_score(y_test, predict_test)
        self.auc = metrics.roc_auc_score(y_test, predict_test)
        self.f1_score = metrics.f1_score(y_test, predict_test)
        self.cm = confusion_matrix(y_test, predict_test)

    def print(self):
        print("Accuracy\tPrecision\tRecall\tAUC\tF1")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (self.acccuracy, self.preceission, self.recall, self.auc, self.f1_score))

        print("Confusion Matrix")
        print(self.cm)

def normalize_value(df):
    max_val = np.percentile(df["value"], [90])[0]
    df["value"] = df["value"] / max_val
    return df



def residual_without_seasonalitiy_via_acorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag - 1]
    if np.abs(r) > 0.5 and lag > 5:
        return x[-1] - x[-lag-1]
    else:
        return x[-1]







def load_featurize_univeriate_data(file_list, value_feature_name, test_fraction=0.3):
    data_sets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_value(df) for df in data_sets]
    transformed_datasets = [create_timeseries_features_round2(df) for df in normalized_datasets]
    data_set = pd.concat(transformed_datasets)

    data_set = data_set.fillna(0)
    X = pd.DataFrame({"value":data_set[value_feature_name]})
    y = data_set.pop("is_anomaly")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=123,
                                                        shuffle=False)
    return X_train, X_test, y_train, y_test

def under_sample_with_SMOTE(X, y):
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


def build_univairiate_model(file_list, value_feature_name, test_fraction=0.3):
    X_train, X_test, y_train, y_test = load_featurize_univeriate_data(file_list, value_feature_name, test_fraction=test_fraction)
    #
    print("Before Shuffle", X_train.shape)
    print(y_train.value_counts())

    X_train, y_train = under_sample_with_SMOTE(X_train, y_train)

    detector = UnivariateAnomalyDetector()
    detector.train(X_train, X_test, y_train, y_test)
    return detector


def adjust_predictions4nighbourhood(y_test, predict_test, slack=5):
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

class UnivariateAnomalyDetector:
    def __init__(self, filename=None):
        if filename is not None:
            self.clf = pickle.load(open(filename, 'rb'))
        else:
            self.clf = None

    #for now we need features to be generated before calling this
    def train(self, X_train, X_test, y_train, y_test, debug=False):
        params = {"scale_pos_weight": 0.05}
        params['learning_rate'] = 0.001
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_test, label=y_test)

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
        self.clf = lgb.train(params=params, train_set=d_train, num_boost_round=10000, valid_sets=d_test,
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
        adjust_predictions = adjust_predictions4nighbourhood(y_test, predict_test)
        adj_eval = Evaluation(y_test, adjust_predictions)
        print("Adjested Evalauations")
        adj_eval.print()
        return predict_train, predict_test

    def save_model(self, filename):
        pickle.dump(self.clf, open(filename, 'wb'))

    def predict(self, X_test):
        predict_test =  self.clf.predict(X_test)
        return  [1 if v > 0.75 else 0 for v in predict_test]
