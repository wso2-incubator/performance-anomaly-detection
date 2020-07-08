from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random

class ExperimentData:
    def __init__(self, experiment, X_train, X_test, y_train, y_test):
        self.experiment = experiment
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_copy(self):
        return ExperimentData(self.X_train.copy(), self.X_test.copy(), self.y_train, self.y_test)

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


def do_detection(experiment_list, dector_fns):
    for dector_fn in dector_fns:
        predict_train_list = []
        predict_test_list = []
        for e in experiment_list:
            print(e.experiment)
            predict_train, predict_test = dector_fn(e.X_train, e.X_test, e.y_train, e.y_test)
            acccuracy, preceission, recall, auc, f1_score = print_evaluations(e.experiment, e.y_train, e.y_test, predict_train, predict_test)
            predict_test_list.append(predict_test)
            predict_train_list.append(predict_train)

        y_train_all = np.concatenate([e.y_train for e in experiment_list])
        y_test_all = np.concatenate([e.y_test for e in experiment_list])
        acccuracy, preceission, recall, auc, f1_score = print_evaluations("final results",
                                                                        y_train_all, y_test_all, np.concatenate(predict_train_list),
                                                                          np.concatenate(predict_test_list))
        resultsdf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "F1"],
                                 data=[[acccuracy, preceission, recall, auc, f1_score]])
        print(resultsdf)


def print_evaluations(experiment_name, y_train, y_test, predict_train, predict_test):
    print("\n", experiment_name, "Results")
    acccuracy = metrics.accuracy_score(y_test, predict_test)
    preceission = metrics.precision_score(y_test, predict_test)
    recall = metrics.recall_score(y_test, predict_test)
    try:
        auc = metrics.roc_auc_score(y_test, predict_test)
    except:
        auc = 0

    f1_score = metrics.f1_score(y_test, predict_test)

    resultsdf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "F1"], data=[[acccuracy, preceission, recall, auc, f1_score]])
    print(resultsdf)


    print("Confusion Matrix")
    cm = confusion_matrix(y_test, predict_test)
    print(cm)

    return acccuracy, preceission, recall, auc, f1_score

def run_azure_detector(X_train, X_test, y_train, y_test):
    #use latest detector
    #you can train all X_train at once
    #evalaute X_test one by one
    #save all azure results to a csv file in case we need them

    #if you can't return predict_train, then return zeros

    predict_train = pd.Series([random.randint(0, 1) for i in range(len(y_train))])
    predict_test = pd.Series([random.randint(0, 1) for i in range(len(y_test))])
    return predict_train, predict_test


def run_adtk_detector(X_train, X_test, y_train, y_test):
    #use latest detector
    #you can train all X_train at once
    #evalaute X_test one by one
    predict_train = pd.Series([random.randint(0, 1) for i in range(len(y_train))])
    predict_test = pd.Series([random.randint(0, 1) for i in range(len(y_test))])
    return predict_train, predict_test