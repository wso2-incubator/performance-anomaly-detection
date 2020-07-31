# from sklearn import metrics
# from sklearn.metrics import *
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# import random
# from wso2tools.time_series import *
#
# class ExperimentData:
#     def __init__(self, experiment, X_train, X_test, y_train, y_test):
#         self.experiment = experiment
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#
#     def create_copy(self):
#         return ExperimentData(self.X_train.copy(), self.X_test.copy(), self.y_train, self.y_test)
#
#     def save(self):
#         X_train = self.X_train.copy()
#         X_train["target"] = self.y_train.values
#         X_train.to_csv("dm.train.csv")
#
#         X_test = self.X_test.copy()
#         X_test["target"] = self.y_test.values
#         X_test.to_csv("dm.test.csv")
#
#     def save_with_prediction(self, train_predict, test_predict):
#         X_train = self.X_train.copy()
#         X_train["target"] = self.y_train.values
#         X_train["predict"] = train_predict
#         X_train.to_csv("dm.train.csv")
#
#         X_test = self.X_test.copy()
#         X_test["target"] = self.y_test.values
#         X_test["predict"] = test_predict
#         X_test.to_csv("dm.test.csv")
#
#
# def do_detection(experiment_list, dector_fns):
#     for dector_fn in dector_fns:
#         predict_train_list = []
#         predict_test_list = []
#         for e in experiment_list:
#             print(e.experiment)
#             predict_train, predict_test = dector_fn(e.X_train, e.X_test, e.y_train, e.y_test)
#             acccuracy, preceission, recall, auc, f1_score = print_evaluations(e.experiment, e.y_train, e.y_test, predict_train, predict_test)
#             predict_test_list.append(predict_test)
#             predict_train_list.append(predict_train)
#
#         y_train_all = np.concatenate([e.y_train for e in experiment_list])
#         y_test_all = np.concatenate([e.y_test for e in experiment_list])
#         acccuracy, preceission, recall, auc, f1_score = print_evaluations("final results",
#                                                                         y_train_all, y_test_all, np.concatenate(predict_train_list),
#                                                                           np.concatenate(predict_test_list))
#         resultsdf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "F1"],
#                                  data=[[acccuracy, preceission, recall, auc, f1_score]])
#         print(resultsdf)
#
#
# def print_evaluations(experiment_name, y_train, y_test, predict_train, predict_test):
#     print("\n", experiment_name, "Results")
#     acccuracy = metrics.accuracy_score(y_test, predict_test)
#     preceission = metrics.precision_score(y_test, predict_test)
#     recall = metrics.recall_score(y_test, predict_test)
#     try:
#         auc = metrics.roc_auc_score(y_test, predict_test)
#     except:
#         auc = 0
#
#     f1_score = metrics.f1_score(y_test, predict_test)
#
#     resultsdf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "F1"], data=[[acccuracy, preceission, recall, auc, f1_score]])
#     print(resultsdf)
#
#
#     print("Confusion Matrix")
#     cm = confusion_matrix(y_test, predict_test)
#     print(cm)
#
#     return acccuracy, preceission, recall, auc, f1_score
#
# def run_azure_detector(X_train, X_test, y_train, y_test):
#     #use latest detector
#     #you can train all X_train at once
#     #evalaute X_test one by one
#     #save all azure results to a csv file in case we need them
#
#     #if you can't return predict_train, then return zeros
#
#     predict_train = pd.Series([random.randint(0, 1) for i in range(len(y_train))])
#     predict_test = pd.Series([random.randint(0, 1) for i in range(len(y_test))])
#     return predict_train, predict_test
#
#
# def run_adtk_detector(X_train, X_test, y_train, y_test):
#     #use latest detector
#     #you can train all X_train at once
#     #evalaute X_test one by one
#     predict_train = pd.Series([random.randint(0, 1) for i in range(len(y_train))])
#     predict_test = pd.Series([random.randint(0, 1) for i in range(len(y_test))])
#     return predict_train, predict_test
#
# def normalize_value(df):
#     max_val = np.percentile(df["value"], [90])[0]
#     df["value"] = df["value"] / max_val
#     return df
#
#
#
# def run_models(model, X_train, X_test, y_train, y_test):
#     if model == "xgb":
#         print("before XGB", collections.Counter(y_train))
#         target_value_counts = y_train.value_counts()
#         class_weight = target_value_counts[0] / target_value_counts[1]
#         print("class_weight", class_weight)
#         '''
#         xgb_params = {'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc'],
#                       'eta': 0.05,
#                       'max_depth': 3,
#                       'scale_pos_weight': 0.25,
#                       #'scale_pos_weight': class_weight / 32,  # need this for simple over sampling
#                       # 'scale_pos_weight': class_weight, # use this for smote
#                       # 'min_child_weight':5
#                       }
#         '''
#         xgb_params = {'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc'],
#                       'eta': 0.1,
#                       'max_depth': 7,
#                       'scale_pos_weight': 0.01, #default
#                       #'scale_pos_weight': 0.1,
#                       #'scale_pos_weight': class_weight / 32,  # need this for simple over sampling
#                       #'scale_pos_weight': class_weight,  # use this for smote
#                       # 'min_child_weight':5
#                       }
#
#         predict_train, predict_test = run_xgboost(X_train, X_test, y_train, y_test,
#                                                   use_cv=True, xgb_params=xgb_params)
#     elif model == "RNN":
#         predict_train, predict_test, y_train, y_test = run_1drnn(X_train["value"], X_test["value"], y_train, y_test)
#     elif model == "TF":
#         predict_train, predict_test = run_tf_classification_ublanaced_data(X_train, X_test, y_train, y_test, epochs=20)
#     elif model == "Ridge":
#         predict_train, predict_test = run_ride_classification_for_unbalanced_ts(X_train, X_test, y_train, y_test)
#     elif model == "IForest":
#         predict_train, predict_test = run_isolation_forest(X_train, X_test, y_train, y_test)
#     elif model == "lgbm":
#         #params = {"is_unbalance":True}
#         params = {"scale_pos_weight": 0.05}
#         params['learning_rate'] = 0.001
#         predict_train, predict_test = run_lgbm(X_train, X_test, y_train, y_test, params)
#         #predict_train, predict_test = run_lgbm_cv(X_train, X_test, y_train, y_test, params)
#         predict_train, predict_test = np.round(predict_train), np.round(predict_test)
#     else:
#         raise Exception("unknown model ", model)
#     eval = Evaluation(y_test, predict_test)
#     eval.print()
#
#
#     #we save results
#     X_test = X_test.copy()
#     X_test["actual"] = y_test
#     X_test["predicted"] = predict_test
#     X_test.to_csv("forValidation.csv")
#
#     adjust_predictions = adjust_predictions4nighbourhood(y_test, predict_test)
#     adj_eval = Evaluation(y_test, adjust_predictions)
#     print("Adjested Evalauations")
#     adj_eval.print()
#
#     return eval, adj_eval
#
# def load_featurize_univeriate_data(file_list, value_feature_name):
#     data_sets = [pd.read_csv(f) for f in file_list]
#     normalized_datasets = [normalize_value(df) for df in data_sets]
#     transformed_datasets = [create_timeseries_features_round2(df) for df in normalized_datasets]
#     data_set = pd.concat(transformed_datasets)
#
#     data_set = data_set.fillna(0)
#     X = pd.DataFrame({"value":data_sets[value_feature_name]})
#     y = data_set.pop("is_anomaly")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123,
#                                                         shuffle=False)
#     return X_train, X_test, y_train, y_test
#
# def build_univairiate_model(file_list, value_feature_name):
#     X_train, X_test, y_train, y_test = load_featurize_univeriate_data(file_list, value_feature_name)
#     #
#     print("Before Shuffle", X_train.shape)
#     print(y_train.value_counts())
#
#     # X_train, y_train = over_sample(X_train, y_train)
#     # X_train, y_train = over_sample_with_smote(X_train, y_train)
#     X_train, y_train = under_sample_with_SMOTE(X_train, y_train)
#     # print("After Shuffle", X_train.shape)
#     # print(y_train.value_counts())
#
#     # models = ["xgb", "RNN", "lgbm"]
#
#     models = ["lgbm"]
#     results = []
#     for m in models:
#         eval, adj_eval = run_models(m, X_train, X_test, y_train, y_test)