import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import lightgbm as lgb

print(tf.__version__)

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow import keras


from sklearn.ensemble import IsolationForest

from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pandas as pd
import category_encoders as ce
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import imblearn
import collections




class Data4Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_copy(self):
        return Data4Model(self.X_train.copy(), self.X_test.copy(), self.y_train, self.y_test)

    def save(self):
        X_train = self.X_train.copy()
        X_train["target"] = self.y_train.values
        X_train.to_csv("dm.train.csv")

        X_test = self.X_test.copy()
        X_test["target"] = self.y_test.values
        X_test.to_csv("dm.test.csv")

    def save_with_prediction(self, train_predict, test_predict):
        X_train = self.X_train.copy()
        X_train["target"] = self.y_train.values
        X_train["predict"] = train_predict
        X_train.to_csv("dm.train.csv")

        X_test = self.X_test.copy()
        X_test["target"] = self.y_test.values
        X_test["predict"] = test_predict
        X_test.to_csv("dm.test.csv")


def run_xgboost(X_train, X_test, y_train, y_test, xgb_params=None, use_cv=False, num_rounds=5):
    if xgb_params is None:
        xgb_params = {'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc'],
                      'eta': 0.1,
                      'max_depth': 5}

    print("xgb_params=", xgb_params)
    train_data = xgb.DMatrix(X_train, label=y_train, missing=float('nan'))
    test_data = xgb.DMatrix(X_test, y_test, missing=float('nan'))
    evallist = [(test_data, 'eval'), (train_data, 'train')]
    evalresults_dic = {}

    if use_cv:
        #for time series we keep the shuffle false
        cvresult = xgb.cv(xgb_params, train_data, num_boost_round=500, nfold=5,
                          metrics={'auc'}, verbose_eval=True, early_stopping_rounds=10, shuffle=True)
        print(cvresult)
        num_rounds = len(cvresult)
    gbdt = None
    print("retraining with ", num_rounds)
    # gbdt = xgb.train( xgb_params, train_data, num_rounds, evallist, verbose_eval = True, early_stopping_rounds=5)
    gbdt = xgb.train(xgb_params, train_data, num_rounds, evallist, verbose_eval=True, evals_result=evalresults_dic, )

    fmap = gbdt.get_score(importance_type='total_gain')
    print(fmap.values())

    featureDf = pd.DataFrame({'Features': list(fmap.keys()), 'Importance': list(fmap.values())}).sort_values(
        by='Importance',
        ascending=False)
    print(featureDf)
    print("Features In Order", featureDf["Features"])
    # print("using params ", xgb.get_xgb_params())
    # print("using params ", xgb.get_params())

    # ceate_feature_map_for_feature_importance(features)
    # show_feature_importance(gbdt, feature_names=features)

    predict_test = gbdt.predict(xgb.DMatrix(X_test, missing=float("nan")))
    predict_train = gbdt.predict(xgb.DMatrix(X_train, missing=float("nan")))

    plot_precision_recall(y_test, predict_test)


    return np.round(predict_train), np.round(predict_test)


def run_random_forest(X_train, X_test, y_train, y_test, class_weight=None):
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=1000, class_weight=class_weight)
    clf.fit(X_train, y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    return predict_train, predict_test


def run_ride_classification(X_train, X_test, y_train, y_test):
    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    return predict_train, predict_test


def run_isolation_forest(X_train, X_test, y_train, y_test):
    #clf = IsolationForest(max_samples=100, contamination=0.03, random_state=112)
    #clf = IsolationForest(max_samples=100, contamination=0.1, random_state=112) #baysian values optimize auc
    clf = IsolationForest(max_samples=50, contamination=0.03, random_state=112)  # baysian values optimize f1
    clf.fit(X_train, y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    print(np.percentile(predict_train, [25, 50, 75, 99]))
    print(predict_train)
    return [1 if v == -1 else 0 for v in predict_train], [1 if v == -1 else 0 for v in predict_test]




def run_ride_classification_for_unbalanced_ts(X_train, X_test, y_train, y_test):
    target_value_counts = y_train.value_counts()
    print(target_value_counts)
    class_weight = target_value_counts[0]/target_value_counts[1]
    clf = RidgeClassifier(class_weight={0:1, 1:class_weight}, fit_intercept=True, normalize=True)
    print(clf)
    clf.fit(X_train, y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    return predict_train, predict_test

'''
def run_tf_classification_simple(X_train, X_test, y_train, y_test, epochs=1000, handle_unbalanced=False):
    layers_list = []
    for i in range(2):
        layers_list.append(layers.Dense(256, activation='relu'))
    layers_list.append(layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential(layers_list)

    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  # loss is for update the model during backpropagation
                  # loss=tf.keras.metrics.AUC(),
                  metrics=['AUC'])  # can use anything from matrics here
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    X_train, X_test = normalize(X_train, X_test)

    early_history = model.fit(X_train.values, y_train.values, epochs=epochs, validation_split=0.3, verbose=1,
                              callbacks=[early_stop])
    print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))

    predict_train = np.round(model.predict(X_train))
    predict_test = np.round(model.predict(X_test))
    return predict_train, predict_test
'''


def run_tf_classification(X_train, X_test, y_train, y_test, epochs=1000, handle_unbalanced=False):
    output_bias = None

    layers_list = []
    for i in range(5):
        layers_list.append(layers.Dense(256, activation='relu'))
        layers_list.append(layers.Dropout(0.1))

    layers_list.append(layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential(layers_list)

    # https://keras.io/optimizers/
    # optimizer = 'adam'
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    # optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

    # model.compile(optimizer='adam',
    '''
    see https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    most classifications, what works is BinaryCrossentropy
    '''
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  # loss is for update the model during backpropagation
                  # loss=tf.keras.metrics.AUC(),
                  metrics=['AUC'])  # can use anything from matrics here

    '''
    model = tf.keras.Sequential([
      layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        #layers.Softmax()
    ])
    model.compile(optimizer='adam',
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
          metrics=['accuracy'])


    '''


    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    X_train, X_test = normalize(X_train, X_test)

    early_history = model.fit(X_train.values, y_train.values, epochs=epochs, validation_split=0.3, verbose=1,
                              callbacks=[early_stop])

    '''
    use Adam 
    class_weight = {0: weight_for_0, 1: weight_for_1}
    in fit (    class_weight=class_weight) 

    '''

    print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))

    predict_train = np.round(model.predict(X_train))
    predict_test = np.round(model.predict(X_test))
    return predict_train, predict_test



def run_tf_classification_ublanaced_data(X_train, X_test, y_train, y_test, epochs=1000):
    print("Using tensorflow")
    target_value_counts = y_train.value_counts()
    pos_count = target_value_counts[1]
    neg_count = target_value_counts[0]
    initial_bias = np.log([pos_count / neg_count])
    output_bias = tf.keras.initializers.Constant(initial_bias)

    total = neg_count + pos_count
    weight_for_0 = (1 / neg_count) * (total) / 2.0
    weight_for_1 = (1 /pos_count) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    ##class_weight = {0: 1, 1: 10}
    #class_weight = {0: 1, 1: 1}

    layers_list = []
    for i in range(5):
        #use tanh or leaky relu if all zeros - https://github.com/keras-team/keras/issues/3687
        #layers_list.append(layers.Dense(512, activation='relu'))
        #layers_list.append(layers.Dense(256, activation='tanh'))
        layers_list.append(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        #layers_list.append(layers.Dropout(0.1))
    layers_list.append(layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))


    model = tf.keras.Sequential(layers_list)

    #since we use classweight, best to use Adam - Note: Using class_weights changes the range of the loss. This may affect the stability of the training depending on the optimizer.
    # Optimizers whose step size is dependent on the magnitude of the gradient, like optimizers.SGD, may fail.
    # The optimizer used here, optimizers.Adam, is unaffected by the scaling change. Also note that because of the weighting,
    # the total losses are not comparable between the two models.
    optimizer = keras.optimizers.Adam(lr=1e-4)

    #optimizer = keras.optimizers.Adam()
    #optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    '''
    see https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    most classifications, what works is BinaryCrossentropy
    '''
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  # loss is for update the model during backpropagation
                  # loss=tf.keras.metrics.AUC(),
                  metrics=['Precision', "AUC"])  # can use anything from matrics here
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    X_train, X_test = normalize(X_train, X_test)

    print("class_weight", class_weight)
    early_history = model.fit(X_train.values, y_train.values, epochs=epochs, validation_split=0.3, verbose=1,
                              callbacks=[early_stop], class_weight=class_weight)
    print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))



    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)

    plot_precision_recall(y_test, predict_test)


    return np.round(predict_train), np.round(predict_test)

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



from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

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


def over_sample(X_train, y_train):
    df = X_train.copy()
    df["target"] = y_train

    zeros = df[df["target"] == 0]
    ones = df[df["target"] == 1]
    ratio = len(zeros)/ len(ones)

    oversamples_dataset = pd.concat( [zeros] + [ones for i  in range(round(ratio/2))])
    oversamples_dataset = oversamples_dataset.sample(frac=1)
    y_train = oversamples_dataset.pop("target")
    return oversamples_dataset, y_train
import time
import math
def over_sample_with_smote(X_train, y_train):
    df = X_train.copy()
    df["target"] = y_train

    zeros = df[df["target"] == 0]
    ones = df[df["target"] == 1]

    print("before smote", collections.Counter(y_train))
    split_size = 100000
    zero_split_count = math.ceil(zeros.shape[0]/split_size)
    one_split_size = math.ceil(ones.shape[0] / zero_split_count)
    resampled_dfs = []
    for i in range(zero_split_count):
        zero_cutoff = min((i+1)*split_size, zeros.shape[0])
        one_cutoff = min((i + 1) * one_split_size, ones.shape[0])
        X_zero = zeros[i*split_size:zero_cutoff]
        X_one = ones[i*one_split_size:one_cutoff]
        X = pd.concat([X_zero, X_one])
        y = X.pop("target")
        print("feeding smote", collections.Counter(y))
        oversample = imblearn.over_sampling.SMOTE()
        X, y = oversample.fit_resample(X, y)
        X["target"] = y
        resampled_dfs.append(X)
    oversamples_dataset = pd.concat( resampled_dfs)
    oversamples_dataset = oversamples_dataset.sample(frac=1)
    y_train = oversamples_dataset.pop("target")
    print("after smote", collections.Counter(y_train))
    return oversamples_dataset, y_train


# TODO fix

def normalize(X_train, X_test):
    for f in list(X_train):
        mean = X_train[f].mean()
        std = X_train[f].std()
        X_train[f] = (X_train[f] - mean) / std
        X_test[f] = (X_test[f] - mean) / std
    return X_train, X_test


# TODO LGBM
'''
#https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
'''


def run_lgbm(X_train, X_test, y_train, y_test):
    import lightgbm as lgb
    d_train = lgb.Dataset(X_train, label=y_train)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    clf = lgb.train(params, d_train, 100)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    return predict_train, predict_test


def rm_zero(v):
    return v if v != 0 else 0.01


def describe_df(df, feilds):
    for f in feilds:
        print(df[f].describe())


def replace_inf_max(df, cols):
    for c in cols:
        m = df.loc[df[c] != np.inf, c].max()
        df[c].replace(np.inf, m, inplace=True)
    return df


def create_cat_diffs(X_train, X_test, cols, num_feild):
    for c in cols:
        group_mean = X_train.groupby(c)[num_feild].agg(["mean"]).reset_index()
        X_train = pd.merge(X_train, group_mean, left_on=[c], right_on=[c], how="left")
        X_train[c + "_cat_diff"] = X_train[num_feild] - X_train["mean"]
        print(list(X_train), list(group_mean))
        X_train = X_train.drop(["mean"], axis=1)

        X_test = pd.merge(X_test, group_mean, left_on=[c], right_on=[c], how="left")
        X_test[c + "_cat_diff"] = X_test[num_feild] - X_test["mean"]
        X_test = X_test.drop(["mean"], axis=1)
    return X_train, X_test


class ModelResult:
    def __init__(self, name, predict_train, predict_test):
        self.name = name
        self.predict_train = predict_train
        self.predict_test = predict_test


def run_median_model(model_results):
    # print([r.predict_train.reshape(len(r.predict_train)) for r in model_results])
    predict_train = np.median(np.array([r.predict_train.reshape(len(r.predict_train)) for r in model_results]), axis=0)
    predict_test = np.median([r.predict_test.reshape(len(r.predict_test)) for r in model_results], axis=0)
    return np.round(predict_train), np.round(predict_test)


def run_knn_model(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    predict_train = np.round(clf.predict(X_train))
    predict_test = np.round(clf.predict(X_test))
    return predict_train, predict_test


def derive_feild(X_train, X_test, name, feild1, feild2, fn):
    X_train[name] = fn(X_train[feild1], X_train[feild2])
    X_test[name] = fn(X_test[feild1], X_test[feild2])
    return X_train, X_test


def compare_train_test(X_train, X_test, target_feild):
    print("Feature diff Done\n===================")
    flist = list(X_test)
    for f in flist:
        feature_diff = set(X_test[f].unique()) - set(X_train[f].unique())
        if pd.api.types.is_numeric_dtype(X_test[f]) and len(X_train[f].unique()) > 10:
            print(f, " train (", X_train[f].min(), X_train[f].max(), ") test(", X_test[f].min(), X_test[f].max(), ")")
        elif len(feature_diff) > 0:
            print("Feature diff", f, str(list(feature_diff)))
    print("Feature diff Done")
    print("value_cof, unts")
    print(X_train[target_feild].value_counts())
    # print(X_test[target_feild].value_counts())


def encode_columns(data4model, one_hot_cols=[], embedding_cols=[], target_cols=[], woe_cols=[], sum_cols=[],
                   hash_cols=[]):
    if len(target_cols) > 0:
        encoder = ce.TargetEncoder(cols=target_cols)
        encoder.fit(data4model.X_train, data4model.y_train)
        data4model.X_train = encoder.transform(data4model.X_train)
        data4model.X_train = data4model.X_train.fillna(0)

        data4model.X_test = encoder.transform(data4model.X_test)
        data4model.X_test = data4model.X_test.fillna(0)

    if len(woe_cols) > 0:
        encoder = ce.WOEEncoder(cols=woe_cols)
        encoder.fit(data4model.X_train, data4model.y_train)
        data4model.X_train = encoder.transform(data4model.X_train)
        data4model.X_test = encoder.transform(data4model.X_test)
        print("woe applied")
        print(data4model.X_train[woe_cols].head())
        print(data4model.X_train[woe_cols].quantile([0.25, 0.5, 0.75, 0.99]))

    if len(woe_cols) > 0:
        encoder = ce.WOEEncoder(cols=woe_cols)
        encoder.fit(data4model.X_train, data4model.y_train)
        data4model.X_train = encoder.transform(data4model.X_train)
        data4model.X_test = encoder.transform(data4model.X_test)

    if len(hash_cols) > 0:
        encoder = ce.HashingEncoder(cols=hash_cols)
        encoder.fit(data4model.X_train, data4model.y_train)
        data4model.X_train = encoder.transform(data4model.X_train)
        data4model.X_test = encoder.transform(data4model.X_test)

    if len(sum_cols) > 0:
        encoder = ce.BaseNEncoder(cols=sum_cols)
        encoder.fit(data4model.X_train, data4model.y_train)
        data4model.X_train = encoder.transform(data4model.X_train)
        data4model.X_test = encoder.transform(data4model.X_test)
        print("woe applied")
        # print(data4model.X_train[sum_cols].head())
        # print(data4model.X_train[sum_cols].quantile([0.25, 0.5, 0.75, 0.99]))

    for f in one_hot_cols:
        encoded_columns = pd.get_dummies(data4model.X_train[f], prefix=f, drop_first=True)
        encoded_columns[f] = data4model.X_train[f]
        encoded_columns = encoded_columns.drop_duplicates(subset=[f])

        data4model.X_train = pd.merge(data4model.X_train, encoded_columns, on=f, how="left").drop(f, axis=1)
        print(data4model.X_test[f])
        data4model.X_test = pd.merge(data4model.X_test, encoded_columns, on=f, how="left").drop(f, axis=1)

        if len(find_emptystr(data4model.X_test)) > 0:
            print("######", f)
            print(encoded_columns)
            # print(data4model.X_train[f])

        # print(list(data4model.X_train))
        # print(data4model.X_train.head())
        # data4model.X_train = data4model.X_train.join(encoded_columns).drop(f, axis=1)
        # print(list(data4model.X_train))

        # data4model.X_test = data4model.X_test.join(encoded_columns).drop(f, axis=1)
        # print(data4model.X_test.head())
    # print("train", len(data4model.X_train), "test", len(data4model.X_test))

    if check4Nan(data4model):
        raise ValueError('Nans found')

    return data4model

'''
def print_evaluations(data4model, predict_train, predict_test):
    from collections import Counter

    y_test = data4model.y_test
    y_train = data4model.y_train
    print(np.unique(predict_train, return_counts=True), np.unique(y_train, return_counts=True))
    print(np.unique(predict_test, return_counts=True), np.unique(y_test, return_counts=True))

    print("\nResults")
    acccuracy = metrics.accuracy_score(data4model.y_test, predict_test)
    preceission = metrics.precision_score(y_test, predict_test)
    recall = metrics.recall_score(y_test, predict_test)
    print("Accuracy : %.4g" % acccuracy)
    print("Precision : %.4g" % preceission)
    print("Recall : %.4g" % recall)
    print(classification_report(y_test, predict_test))

    cm = confusion_matrix(y_test, predict_test)
    print(cm)

    auc = metrics.roc_auc_score(y_test, predict_test)
    print("AUC ROC/AUC:", auc)
    f1_score = metrics.f1_score(y_test, predict_test)
    print("F1 Score Test: %f" % f1_score)
    print("F1 Score Train: %f" % metrics.f1_score(y_train, predict_train))
    return f1_score, acccuracy, preceission, recall
'''

def checkDf4Nan(df):
    na_df = df[df.isna().any(axis=1)]
    na_row_count = na_df.shape[0]
    if na_row_count > 1:
        na_columns = df.columns[df.isna().any()].tolist()
        print("Nans found")
        print(str(na_row_count), "/", str(df.shape[0]), "na_columns=", na_columns)
    emptyIndexs = find_emptystr(df)
    if (len(emptyIndexs) > 1):
        print("Empty sells found", emptyIndexs)


def check4Nan(data4model, verbose=False):
    to_check = [data4model.X_train, data4model.X_test]
    to_check_labels = ["X_train", "X_test"]

    has_nan = False
    has_empty = False
    for i, df in enumerate(to_check):
        na_df = df[df.isna().any(axis=1)]
        na_row_count = na_df.shape[0]
        if na_row_count > 1:
            has_nan = True
            na_columns = df.columns[df.isna().any()].tolist()
            print("Nans found")
            print(to_check_labels[i], str(na_row_count), "/", str(df.shape[0]), "na_columns=", na_columns)
            if verbose:
                print(na_df.head())
        emptyIndexs = find_emptystr(df)
        if (len(emptyIndexs) > 1):
            has_empty = True
            print("Empty sells found", emptyIndexs)
    nan_index = np.isnan(data4model.y_train)
    if (nan_index.any()):
        print("Nans found in  y_train", nan_index)
        has_nan = True

    nan_index = np.isnan(data4model.y_test)
    if (nan_index.any()):
        print("Nans found in  y_test", nan_index)
        has_nan = True

    if has_nan or has_empty:
        data4model.X_train.to_csv("X_train.csv")
        data4model.X_test.to_csv("X_test.csv")
        return True
    else:
        print("No Nans found")
        return False


def find_emptystr(df):
    xindex, yindex = np.where(pd.isnull(df))
    indexs = [[xindex[i], yindex[i]] for i in range(len(xindex))]
    # for v in indexs:
    #    print(df[v[0]])

    return indexs


def replace_rare_categories(X_train, X_test, feilds, cutoff=0.001):
    X_train = X_train.copy()
    X_test = X_test.copy()
    other_feild = list(set(list(X_train)) - set(feilds))[0]
    for f in feilds:
        countdf = X_train.groupby(f)[other_feild].agg(["count"]).reset_index()
        countdf = countdf.sort_values(by=["count"], ascending=False)
        cutoff_count = np.sum(countdf["count"].values) * cutoff
        cat_mapping = {}
        for index, row in countdf.iterrows():
            cat_mapping[row[f]] = "other" if row["count"] < cutoff_count else row[f]
        X_train[f] = X_train[f].map(cat_mapping)
        X_test[f] = X_test[f].map(cat_mapping)
        print(X_train.groupby(f)[other_feild].agg(["count"]).reset_index().sort_values(by=["count"], ascending=False))
    return X_train, X_test


def remove_feature(data4model, flist):
    data4model.X_train = data4model.X_train.drop(flist, axis=1)
    data4model.X_test = data4model.X_test.drop(flist, axis=1)


def add_cat_feature_interaction(data4model, f1, f2):
    data4model.X_train = data4model.X_train.copy()
    data4model.X_test = data4model.X_test.copy()

    fname = f1 + "_" + f2
    data4model.X_train[fname] = data4model.X_train[f1].astype(str) + data4model.X_train[f2].astype(str)
    data4model.X_test[fname] = data4model.X_test[f1].astype(str) + data4model.X_test[f2].astype(str)
    return fname


def create_rnn_univariate_data_with_targetdata(dataset, target_data, start_index, end_index, history_steps, target_steps):
    data = []
    labels = []

    start_index = start_index + history_steps
    if end_index is None:
        end_index = len(dataset) - target_steps

    for i in range(start_index, end_index):
        indices = range(i - history_steps, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_steps, 1)))
        labels.append(target_data[i + target_steps -1])
    return np.array(data), np.array(labels)

def create_rnn_univariate_data(dataset, start_index, end_index, history_steps, target_steps):
    data = []
    labels = []

    start_index = start_index + history_steps
    if end_index is None:
        end_index = len(dataset) - target_steps

    for i in range(start_index, end_index):
        indices = range(i - history_steps, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_steps, 1)))
        labels.append(dataset[i + target_steps])
    return np.array(data), np.array(labels)


def prepare4rnn_Nd(dataset, target, start_index, end_index, history_size,
                   target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def train1d_with_rnn(X_train, y_train, X_test, y_test, optimizer='adam'):

    target_value_counts = pd.Series(y_train).value_counts()
    pos_count = target_value_counts[1]
    neg_count = target_value_counts[0]
    initial_bias = np.log([pos_count / neg_count])
    output_bias = tf.keras.initializers.Constant(initial_bias)

    neg_count = neg_count
    total = neg_count + pos_count
    weight_for_0 = (1 / neg_count) * (total) / 2.0
    weight_for_1 = (1 / pos_count) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    print("initialization started")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    train_univariate = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
    print("initialization done")

    #lstm_layer_size = 128
    lstm_layer_size = 256
    # lstm_layer_size = 8
    #dense_size = 256
    dense_size = 128
    dense_layer_count = 2
    #dense_layer_count = 5
    activation_fn = "relu"
    #activation_fn = tf.keras.layers.LeakyReLU(alpha=0.3)
    #optimizer =  keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)


    layers_list = []
    #layers_list.append(tf.keras.layers.LSTM(lstm_layer_size, input_shape=X_train.shape[-2:]))
    #layers_list.append(tf.keras.layers.LSTM(lstm_layer_size, dropout=0.2, input_shape=X_train.shape[-2:]))
    layers_list.append(tf.keras.layers.GRU(lstm_layer_size, dropout=0.3, recurrent_dropout=0.3,
                                           bias_regularizer=tf.keras.regularizers.l1(0.01),
                                           activity_regularizer=tf.keras.regularizers.l1(0.01)))
    for l in range(dense_layer_count):
        layers_list.append(tf.keras.layers.Dense(dense_size, activation=activation_fn))
        layers_list.append(tf.keras.layers.Dropout(0.3))
    #activation
    layers_list.append(tf.keras.layers.Dropout(0.3))
    layers_list.append(tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))
    simple_lstm_model = tf.keras.models.Sequential(layers_list)

    simple_lstm_model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["AUC"])

    EVALUATION_INTERVAL = 200
    EPOCHS = 30

    print("Running RNN")
    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50, class_weight=class_weight)

    train_predict = simple_lstm_model.predict(X_train)
    test_predict = simple_lstm_model.predict(X_test)
    plot_precision_recall(y_test, test_predict)
    return np.round(train_predict), np.round(test_predict), y_train, y_test


def save_dataset2_model(X, y, name="dataset.csv"):
    X = X.copy()
    X["target"] = y
    X.to_csv(name)


def run_1drnn(X_train, X_test, y_train, y_test):
    print(X_train)
    #X_train is a series
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_train.mean()) / X_train.std()

    X_train = X_train.values
    X_test = X_test.values

    univariate_past_history = 20
    univariate_future_target = 0
    print(X_train)
    X_train, y_train = create_rnn_univariate_data_with_targetdata(X_train, y_train.values, 0, X_train.size,
                                                  univariate_past_history,
                                                  univariate_future_target)
    X_test, y_test = create_rnn_univariate_data_with_targetdata(X_test, y_test.values, 0, X_test.size,
                                                univariate_past_history,
                                                univariate_future_target)
    return train1d_with_rnn(X_train, y_train, X_test, y_test)

def test_id():
    df = pd.read_csv("/Users/srinath/playground/Datasets/weather/jena_climate_2009_2016.csv")

    # we will use one feild
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    # uni_data.head()
    # uni_data.plot(subplots=True)
    # plt.show()

    uni_data = uni_data.values

    TRAIN_SPLIT = 300000

    # Let's standardize the data.

    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data - uni_train_mean) / uni_train_std

    univariate_past_history = 20
    univariate_future_target = 0

    X_train, y_train = create_rnn_univariate_data(uni_data, 0, TRAIN_SPLIT,
                                                  univariate_past_history,
                                                  univariate_future_target)
    X_test, y_test = create_rnn_univariate_data(uni_data, TRAIN_SPLIT, None,
                                                univariate_past_history,
                                                univariate_future_target)
    predict_train, predict_test = train1d_with_rnn(X_train, y_train, X_test, y_test)

    '''





    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = create_rnn_univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = create_rnn_univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mape')

    EVALUATION_INTERVAL = 200
    EPOCHS = 20

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)

    test_predict = simple_lstm_model.predict(x_val_uni)
    '''

    forecast_data = pd.DataFrame()
    forecast_data["actual"] = y_test
    forecast_data["predicted"] = predict_test
    forecast_data["index"] = range(len(y_test))

    ax = sns.lineplot(x="index", y="predicted", data=forecast_data)
    ax = sns.lineplot(x="index", y="actual", data=forecast_data, ax=ax)
    plt.show()


'''
for parameters see https://lightgbm.readthedocs.io/en/latest/Parameters.html
'''
def run_lgbm(X_train, X_test, y_train, y_test, params=None):
    d_train = lgb.Dataset(X_train, label=y_train)
    d_test = lgb.Dataset(X_test, label=y_test)
    if params is None:
        params = {}
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
    #clf = lgb.train(params=params, train_set=d_train, num_boost_round=100)
    #clf = lgb.train(params=params, train_set=d_train, num_boost_round=10000, valid_sets=d_test, early_stopping_rounds=1000, verbose_eval=True)
    clf = lgb.train(params=params, train_set=d_train, num_boost_round=10000, valid_sets=d_test,
                    verbose_eval=True)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    importance_df = pd.DataFrame(sorted(zip(clf.feature_importance(), X_train.columns)), columns=['Value', 'Feature'])
    print(importance_df.sort_values(by=['Value'], ascending=False))
    plot_precision_recall(y_test, predict_test)

    X_test = X_test.copy()
    X_test["actual"] = y_test
    X_test["predicted"] = predict_test
    X_test.to_csv("results_prob.csv")


    return predict_train, predict_test

def run_lgbm_cv(X_train, X_test, y_train, y_test, params=None):
    d_train = lgb.Dataset(X_train, label=y_train)
    d_test = lgb.Dataset(X_test, label=y_test)
    if params is None:
        params = {}
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
    #clf = lgb.train(params=params, train_set=d_train, num_boost_round=100)

    eval_hist = lgb.cv(params, train_set=d_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=False, metrics=None,
           fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto',
           early_stopping_rounds=300, fpreproc=None, verbose_eval=True, show_stdv=True, seed=0, callbacks=None,
           eval_train_metric=True)

    print("eval_hist=",eval_hist)
    clf = lgb.train(params=params, train_set=d_train, num_boost_round=len(eval_hist), valid_sets=d_test,
                    verbose_eval=True)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    importance_df = pd.DataFrame(sorted(zip(clf.feature_importance(), X_train.columns)), columns=['Value', 'Feature'])
    print(importance_df.sort_values(by=['Value'], ascending=False))
    plot_precision_recall(y_test, predict_test)

    return predict_train, predict_test


