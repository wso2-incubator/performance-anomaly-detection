'''
https://docs.google.com/spreadsheets/d/16vvjcdDs3UpoUbEJaTyJZRZH3TM54iWsT6PrT32r36E/edit#gid=0

'''
from sklearn.ensemble import IsolationForest

import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from skopt.space import Real, Integer
import itertools
import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *
from wso2tools.model_templates import Data4Model, run_xgboost

dm_global = None
dimensions_global = None

#There are several data types using which you can define the search space. Those are Categorical, Real and Integer. When defining a search space that involves floating point values you should go for “Real” and if it involves integers, go for “Integer”. If your search space involves categorical values like different activation functions, then you should go for the “Categorical” type.


'''
#Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the
# weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
#default 0.3
eta = Real(low=0.05, high=1.0, prior='uniform', name='eta')
#Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
# 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth.
# Beware that XGBoost aggressively consumes memory when training a deep tree.
#default 6
max_depth = Integer(low=2, high=9, name='max_depth')
#scale_pos_weight [default=1] Control the balance of positive and negative weights, useful for unbalanced classes.
# A typical value to consider: sum(negative instances) / sum(positive instances). See Parameters Tuning for more discussion.
# Also, see Higgs Kaggle competition demo for examples: R, py1, py2, py3.
# use negative sample count/ positive sample count [5 in this case]
scale_pos_weight = Integer(low=1, high=9, name='scale_pos_weight')
# Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the
# training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
#default 1
subsample = Real(low=0.5, high=1.0, prior='uniform', name='subsample')
#colsample_bytree, colsample_bylevel, colsample_bynode [default=1] This is a family of parameters for subsampling of columns.
# All colsample_by* parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.
# colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
# colsample_bylevel is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree.
# Columns are subsampled from the set of columns chosen for the current tree.
colsample_bytree = Real(low=0.5, high=1.0, prior='uniform', name='colsample_bytree')
#Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the
#more conservative the algorithm will be.range: [0,∞]
gamma = Real(low=1, high=10000, prior='log-uniform', name='gamma') #0

#min_child_weight [default=1] Minimum sum of instance weight (hessian) needed in a child. If the tree partition step
# results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will
# give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances
# needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
min_child_weight = Integer(low=2, high=9, name='min_child_weight')

dimensions = [eta, max_depth, subsample, colsample_bytree, gamma, min_child_weight]

default_parameters = [0.3, 3, 1, 1, 3, 2]


'''

experiment_results = []

def tune_parameters_with_baysian(dm, dimensions, default_parameters, experiment_fn, experiment_count=100):
    global dm_global
    global dimensions_global
    dm_global = dm
    dimensions_global = dm


    print("starting baysian Optimizations")
    search_result = gp_minimize(func=experiment_fn,
                                   dimensions=dimensions,
                                   acq_func='EI',  # Expected Improvement.
                                   n_calls=experiment_count,
                                   x0=default_parameters,
                                   random_state=46)
    #can use x0, y0 to eariler experiments
    plot = plot_convergence(search_result, yscale="log")
    best_parameters = search_result.x
    best_score = search_result.fun

    results_df = pd.DataFrame(experiment_results, columns=["config", "score"])
    results_df = results_df.sort_values(by=["score"], ascending=False)
    return best_parameters, best_score, results_df


#def run_experiment(eta_step, max_depth_step, subsample_step, colsample_bytree_step, gamma_step, min_child_weight_step):
#@use_named_args(dimensions_global)
#def run_experiment(max_samples, contamination):
def run_experiment(args):
    max_samples = args[0]
    contamination = args[1]
    clf = IsolationForest(max_samples=max_samples, contamination=contamination, random_state=112)
    clf.fit(dm_global.X_train, dm_global.y_train)
    predict_train = clf.predict(dm_global.X_train)
    predict_test = clf.predict(dm_global.X_test)
    predict_train = [1 if v == -1 else 0 for v in predict_train]
    predict_test = [1 if v == -1 else 0 for v in predict_test]

    auc, f1 = calculate_evalauations(dm_global.y_train, dm_global.y_test, predict_train, predict_test)
    print(max_samples, contamination, ":auc=", auc, ", f1=", f1)
    experiment_results.append([str(args), f1])
    return 1 - f1

def calculate_evalauations(y_train, y_test, predict_train, predict_test):
    auc = metrics.roc_auc_score(y_test, predict_test)
    f1 = metrics.f1_score(y_test, predict_test)

    preceission = metrics.precision_score(y_test, predict_test)
    recall = metrics.recall_score(y_test, predict_test)
    auc = metrics.roc_auc_score(y_test, predict_test)
    f1_score = metrics.f1_score(y_test, predict_test)

    print("Accuracy\tPrecision\tRecall\tAUC\tF1")
    print("%.2f\t%.2f\t%.2f\t%.2f" %(preceission, recall, auc, f1_score))

    return auc, f1

def run_tunning(X_train, X_test, y_train, y_test):
    max_samples = [50, 100, 300, 500, 1000]
    contamination = [0.01, 0.02, 0.03, 0.05, 0.10]
    dimensions = [max_samples, contamination]
    default_parameters = [100, 0.03]
    dm = Data4Model(X_train, X_test, y_train, y_test)
    best_parameters, best_score, results_df = tune_parameters_with_baysian(dm, dimensions, default_parameters, run_experiment, experiment_count=30)
    print("best_parameters:max_samples=", best_parameters[0], ",contamination=", best_parameters[1], ">best_score=", best_score)
    results_df.to_csv("experiment_results.csv")



def run_xg_boost_tunning(X_train, X_test, y_train, y_test):
    eta = [0.01,0.05, 0.001, 0.0001]
    max_depth = [2,3,5,7]
    scale_pos_weight = [1.0, 4.0, 8.0, 0.25, 0.1, 0.03]
    dimensions = [eta, max_depth, scale_pos_weight]
    default_parameters = [0.001, 5, 1]
    dm = Data4Model(X_train, X_test, y_train, y_test)
    best_parameters, best_score, results_df = tune_parameters_with_baysian(dm, dimensions, default_parameters, run_xgboost_experiment, experiment_count=30)
    print("best_parameters:eta=", best_parameters[0], ",max_depth=", best_parameters[1], "scale_pos_weight",best_parameters[2], ">best_score=", best_score)
    results_df.to_csv("experiment_results.csv")

def run_xgboost_experiment(args):
    eta = args[0]
    max_depth = args[1]
    scale_pos_weight = args[2]

    xgb_params = {'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc'],
                  'eta': eta,
                  'max_depth': max_depth,
                  'scale_pos_weight': scale_pos_weight,
                  # 'scale_pos_weight': class_weight / 32,  # need this for simple over sampling
                  # 'scale_pos_weight': class_weight, # use this for smote
                  # 'min_child_weight':5
                  }
    predict_train, predict_test = run_xgboost(dm_global.X_train, dm_global.X_test, dm_global.y_train, dm_global.y_test,
                                              use_cv=True, xgb_params=xgb_params)
    auc, f1 = calculate_evalauations(dm_global.y_train, dm_global.y_test, predict_train, predict_test)
    print(eta, max_depth, scale_pos_weight, ":auc=", auc, ", f1=", f1)
    experiment_results.append([str(args), f1])
    return 1 - f1
