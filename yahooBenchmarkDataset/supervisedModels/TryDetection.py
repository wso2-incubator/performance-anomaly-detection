#https://arundo-adtk.readthedocs-hosted.com/en/stable/
from adtk.data import validate_series
from random import shuffle
from wso2tools.plot_templates import *
from wso2.time_series import *
from wso2.anomaly_detection import *
import datetime
from wso2tools.model_templates import *
import adtk


start = datetime.datetime(2020, 1, 1, 0, 0)
start_sec = start.timestamp()


def to_date_time(i):
    return datetime.date.fromtimestamp(start_sec + i*60*60*24)

'''
def forecast_with_adtk():
    s_train = pd.read_csv("/Users/srinath/playground/Datasets/SystemsData/YahooAnomalyDataset/A1Benchmark/real_66.csv")
    s_train["time"] = [to_date_time(ts) for ts in s_train["timestamp"].values]
    s_train.index= pd.to_datetime(s_train["time"])
    s_train = s_train["value"]

    print(s_train.head())
    adtk.data.validate_series(s_train)

    plot(s_train)


    #seasonal_ad = SeasonalAD()
    #anomalies = seasonal_ad.fit_detect(s_train)
    esd_ad = GeneralizedESDTestAD(alpha=0.001)
    anomalies = esd_ad.fit_detect(s_train)
    plot(s_train, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
    plt.show()
'''


def evals2df(list_evals):
    results = []
    for eval in list_evals:
        results.append([ eval.acccuracy, eval.preceission, eval.recall, eval.auc, eval.f1_score])
    return  pd.DataFrame(results, columns=["acccuracy", "preceission", "recall", "auc", "f1_score"])


'''
def print_evaluations(y_train, y_test, predict_train, predict_test):
    print("\nResults")
    acccuracy = metrics.accuracy_score(y_test, predict_test)
    preceission = metrics.precision_score(y_test, predict_test)
    recall = metrics.recall_score(y_test, predict_test)
    auc = metrics.roc_auc_score(y_test, predict_test)
    f1_score = metrics.f1_score(y_test, predict_test)

    print("Accuracy\tPrecision\tRecall\tAUC\tF1")
    print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" %(acccuracy, preceission, recall, auc, f1_score))

    resultsdf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "F1"], data=[[acccuracy, preceission, recall, auc, f1_score]])
    print(resultsdf)


    #print("Accuracy : %.4g" %acccuracy )
    #print("Precision : %.4g" %preceission )
    #print("Recall : %.4g" %recall )
    #print(classification_report(y_test, predict_test))

    print("Confusion Matrix")
    cm = confusion_matrix(y_test, predict_test)
    print(cm)

    #print("AUC ROC/AUC:", auc)
    #print("F1 Score Test: %f" % f1_score)
    #print("F1 Score Train: %f" % metrics.f1_score(y_train, predict_train))
   # ROC curves should be used when there are roughly equal numbers of observations for each class.
   # Precision-Recall curves should be used when there is a moderate to large class imbalance.

    #todo switch to probilities and then generate precisoin recall curve 
    
    #precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    return acccuracy, preceission, recall, auc, f1_score

'''







def calculate_batch_anomaly(list):
    max_limit = len(list)
    result = []
    for i in range(max_limit):
        array = list[i-20:i]
        if(len(array) == 0):
            result.append(0)
        else:
            result.append(0 if np.sum(array) == 0 else 1)
    return result

def copy_org(df):
    df["value.org"] = df["value"]
    return df


def load_given_datasets(feature_fn=create_timeseries_features, file_list=None):
    dir = "/Users/srinath/playground/Datasets/SystemsData/YahooAnomalyDataset/A1Benchmark/"
    data_sets = [pd.read_csv(dir+"real_" +str(i)+".csv") for i in file_list]
    normalized_datasets = [normalize_df_withp90(copy_org(df)) for df in data_sets]
    transformed_datasets = [feature_fn(df) for df in normalized_datasets]
    data_set = pd.concat(transformed_datasets)

    data_set = data_set.fillna(0)
    data_set= data_set.drop(["timestamp"], axis=1)

    X = data_set
    y = data_set.pop("is_anomaly")
    return X, y



def load_yahoo_dataset(feature_fn=create_timeseries_features, file_list=None):
    dir = "/Users/srinath/playground/Datasets/SystemsData/YahooAnomalyDataset/A1Benchmark/"
    if file_list is None:
        file_list = list(range(65))
        shuffle(file_list)
        #print("file_list", file_list)
        file_list = [ 11, 19, 15, 9, 45, 56, 59, 64, 41, 16, 54, 39, 2, 44, 23, 18, 47, 31, 24, 49, 26, 3, 0, 60, 52, 40, 43, 61, 12, 17, 33, 20, 1, 38, 7, 50, 30, 32, 21, 48, 5, 35, 4, 14, 29, 22, 55, 8, 62, 6, 58, 36, 53, 34, 10, 63, 28, 51, 25, 42, 57, 46, 13, 37, 27]

    data_sets = [pd.read_csv(dir+"real_" +str(i+1)+".csv") for i in file_list]
    normalized_datasets = [normalize_df_withp90(df) for df in data_sets]
    transformed_datasets = [feature_fn(df) for df in normalized_datasets]
    data_set = pd.concat(transformed_datasets)

    data_set = data_set.fillna(0)
    data_set= data_set.drop(["timestamp"], axis=1)
    #data_set["is_anomaly"] = calculate_batch_anomaly(data_set["is_anomaly"].values)

    y = data_set.pop("is_anomaly")
    X_train, X_test, y_train, y_test = train_test_split(data_set, y, test_size=0.3, random_state=123,
                                                        shuffle=False)
    return X_train, X_test, y_train, y_test


def feature_based_forecast():
    X_train, X_test, y_train, y_test = load_yahoo_dataset(create_timeseries_features_round2)
    #
    print("Before Shuffle", X_train.shape)
    print(y_train.value_counts())

    #X_train, y_train = over_sample(X_train, y_train)
    #X_train, y_train = over_sample_with_smote(X_train, y_train)
    X_train, y_train = under_sample_with_SMOTE(X_train, y_train)
    #print("After Shuffle", X_train.shape)
    #print(y_train.value_counts())

    #models = ["xgb", "RNN", "lgbm"]

    models = ["lgbm"]
    results =[]
    for m in models:
         eval, adj_eval = run_models(m, X_train, X_test, y_train, y_test)


def run_fold(train_data_sets, test_data_sets):
    X_train, y_train = load_given_datasets(create_timeseries_features_round2, file_list=train_data_sets)
    X_test, y_test = load_given_datasets(create_timeseries_features_round2, file_list=test_data_sets)

    eval, adj_eval = run_models("lgbm", X_train, X_test, y_train, y_test)

    return eval, adj_eval


def kfold_forecast():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    X = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
         59, 60, 61, 62, 63, 64, 65, 66, 67])
    # y contains the labels for each type of data set based on the anomalies they carry
    # 0 - spikes only, 1 - level shifts only, 2 - dips only, 3 - mixed anomalies, 4 - other types of anomalies (e.g. violation/change of seasonality), 5 - no anomalies
    y = np.array(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 0, 1, 2, 3, 1, 4, 3, 3, 2, 0, 3, 3, 0, 3, 5, 2, 3,
         2, 3, 3, 0, 0, 3, 3, 0, 3, 4, 1, 2, 0, 2, 2, 3, 3, 2, 2, 4, 1, 5, 0, 3, 0, 1, 5, 0, 0, 0])

    # Stratified K fold implementation
    print("Stratified K fold implementation")
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)
    print(skf)

    results = []
    adj_results = []
    fold = 1
    for train_index, test_index in skf.split(X, y):
        print(fold, "TRAIN:", train_index, "TEST:", test_index)
        train_data_sets, test_data_sets = X[train_index], X[test_index]
        eval, adj_eval = run_fold(train_data_sets, test_data_sets)
        results.append(eval)
        adj_results.append(adj_eval)
        #y_train, y_test = y[train_index], y[test_index]
        fold = fold +1

    print("Results")
    resultsDf = evals2df(results)
    adj_resultsDf = evals2df(adj_results)

    resultsDf["type"] = "original"
    adj_resultsDf["type"] = "adjusted"
    pd.concat([resultsDf, adj_resultsDf]).to_csv("kfold_results.csv")

    feilds = ['acccuracy', 'preceission', 'recall', 'auc', 'f1_score']
    df = pd.read_csv("kfold_results.csv")

    df_org = df[df["type"] == "original"]
    print("Original\n=======")
    for f in feilds:
        # print(df_org[f].describe())
        print(f, "mean=%.2f, std=%.2f, min=%.2f, max=%.2f" % (
        df_org[f].mean(), df_org[f].std(), df_org[f].min(), df_org[f].max()))

    print("Adjusted\n====")
    df_adj = df[df["type"] == "adjusted"]
    for f in feilds:
        print(f, "mean=%.2f, std=%.2f, min=%.2f, max=%.2f" % (
        df_adj[f].mean(), df_adj[f].std(), df_adj[f].min(), df_adj[f].max()))


def run_models(model, X_train, X_test, y_train, y_test):
    if model == "xgb":
        print("before XGB", collections.Counter(y_train))
        target_value_counts = y_train.value_counts()
        class_weight = target_value_counts[0] / target_value_counts[1]
        print("class_weight", class_weight)
        '''
        xgb_params = {'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc'],
                      'eta': 0.05,
                      'max_depth': 3,
                      'scale_pos_weight': 0.25,
                      #'scale_pos_weight': class_weight / 32,  # need this for simple over sampling
                      # 'scale_pos_weight': class_weight, # use this for smote
                      # 'min_child_weight':5
                      }
        '''
        xgb_params = {'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc'],
                      'eta': 0.1,
                      'max_depth': 7,
                      'scale_pos_weight': 0.01, #default
                      #'scale_pos_weight': 0.1,
                      #'scale_pos_weight': class_weight / 32,  # need this for simple over sampling
                      #'scale_pos_weight': class_weight,  # use this for smote
                      # 'min_child_weight':5
                      }

        predict_train, predict_test = run_xgboost(X_train, X_test, y_train, y_test,
                                                  use_cv=True, xgb_params=xgb_params)
    elif model == "RNN":
        predict_train, predict_test, y_train, y_test = run_1drnn(X_train["value"], X_test["value"], y_train, y_test)
    elif model == "TF":
        predict_train, predict_test = run_tf_classification_ublanaced_data(X_train, X_test, y_train, y_test, epochs=20)
    elif model == "Ridge":
        predict_train, predict_test = run_ride_classification_for_unbalanced_ts(X_train, X_test, y_train, y_test)
    elif model == "IForest":
        predict_train, predict_test = run_isolation_forest(X_train, X_test, y_train, y_test)
    elif model == "lgbm":
        #params = {"is_unbalance":True}
        params = {"scale_pos_weight": 0.05}
        params['learning_rate'] = 0.001
        predict_train, predict_test = run_lgbm(X_train, X_test, y_train, y_test, params)
        #predict_train, predict_test = run_lgbm_cv(X_train, X_test, y_train, y_test, params)
        predict_train, predict_test = np.round(predict_train), np.round(predict_test)
    else:
        raise Exception("unknown model ", model)
    eval = Evaluation(y_test, predict_test)
    eval.print()


    #we save results
    X_test = X_test.copy()
    X_test["actual"] = y_test
    X_test["predicted"] = predict_test
    X_test.to_csv("forValidation.csv")

    adjust_predictions = adjust_predictions4nighbourhood(y_test, predict_test)
    adj_eval = Evaluation(y_test, adjust_predictions)
    print("Adjested Evalauations")
    adj_eval.print()

    return eval, adj_eval

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
    '''
    changed_indexs = []
    for i in range(length):
        if(adjusted_forecasts[i] != predict_test[i]):
            changed_indexs.append(i)
    print(len(changed_indexs), "changed", changed_indexs)
    '''

    return adjusted_forecasts

'''

def find_best_cutoff_with_neighbourhood_adjustment():
    slack = 20
    results = pd.read_csv("results_prob.csv")
    y_test = results["actual"]
    predict_test = results["predicted"]

    c = collections.Counter(y_test)
    percentile = np.round(100*c[0]/(c[0]+c[1]))
    print(c[0], c[1])
    print(percentile)
    print(np.percentile(predict_test, [percentile]))

    data = []
    for threshold in range(50, 100):
        predict_test_cutoff = [0 if x < threshold/100 else 1 for x in predict_test ]
        adjust_predictions = adjust_predictions4nighbourhood(y_test, predict_test_cutoff, slack=slack)
        _, adj_preceission, adj_recall, _, adj_f1_score = calculate_evalauations(y_test, adjust_predictions)
        data.append([adj_preceission, adj_recall, adj_f1_score, threshold])
        print(threshold, "Adjusted: adj_preceission=", adj_preceission, "adj_recall=", adj_recall, "adj_f1_score=", adj_f1_score)

    resultsDf = pd.DataFrame(data, columns=["adj_preceission", "adj_recall", "adj_f1_score", "threshold"])
    plot_xy([
        (resultsDf["adj_preceission"], resultsDf["adj_recall"], "precision vs. recall"),
        (resultsDf["threshold"], resultsDf["adj_preceission"], "precision"),
        (resultsDf["threshold"], resultsDf["adj_recall"], "recall"),
        (resultsDf["threshold"], resultsDf["adj_f1_score"], "f1"),
        (resultsDf["adj_f1_score"], resultsDf["adj_preceission"], "f1 vs. precision"),
        ], "adj_tradeoff.png")
    resultsDf = resultsDf[resultsDf["adj_preceission"]>=0.75]
    resultsDf.sort_values(by=["adj_preceission"])
    print("Selected", resultsDf.head(1))
    threshold = resultsDf.iloc[ 0 , : ]["threshold"]

    predict_test_cutoff = [0 if x < threshold / 100 else 1 for x in predict_test]
    adjust_predictions = adjust_predictions4nighbourhood(y_test, predict_test_cutoff, slack=slack)

    results["adj_predicted"] = adjust_predictions
    results.to_csv("results_prob.csv")



def vizualize_yahoo_dataset():
    X_train, X_test, y_train, y_test = load_yahoo_dataset()
    #viz_data_2d(X_train, y_train)
    X_train["labels"] = y_train
    ax = sns.scatterplot(x="value", y="value.w50.std", data=X_train, hue="labels")
    #g = sns.jointplot(x="value", y="value.w50.std", data=X_train, kind="kde", hue="labels")

    plt.show()
    #plt.savefig("allseries.png")


def viz_data_2d(X_train, labels):

    X_embedded = TSNE(n_components=2).fit_transform(X_train)
    print(X_embedded.shape)

    df = pd.DataFrame(X_embedded, columns=["f1", "f2"])
    df.to_csv("tsne.csv")
    df["labels"] = labels.values
    df.to_csv("tsne1.csv")


    ax = sns.scatterplot(x="f1", y="f2", data=df, hue="labels")
    #plt.savefig("allseries.png")
    plt.show()

'''




#vizualize_yahoo_dataset()

def test_rnn_data_gen():
    univariate_past_history = 20
    univariate_future_target = 0
    X_train = pd.Series(range(100))
    y_train = pd.Series([2*v for v in range(100)])
    X_train, y_train = create_rnn_univariate_data_with_targetdata(X_train.values, y_train.values, 0, X_train.size,
                                                  univariate_past_history,
                                                  univariate_future_target)
    print(X_train)
    print(y_train)

#test_rnn_data_gen()

def check_cpu_data():
    filename = '/Users/srinath/code/performance-anomaly-detection/yahooBenchmarkDataset/supervisedModels/anomaly_model_lgbm.sav'
    clf_loaded = pickle.load(open(filename, 'rb'))


    cpu_df = pd.read_csv("/Users/srinath/code/my-python-projects/ChoreoAnalysis/choreo-system/cpu-2020.07.04-2020.7.06.csv")
    print(list(cpu_df))
    # ['time', 'container', 'cpu', 'container_short']
    #container = cpu_df["container"].unique()[0]
    #cpu_df = cpu_df[cpu_df["container"] == container]
    #print()

    container_counts = cpu_df.groupby(["container"])["time"].agg(["count"]).reset_index()
    container_counts = container_counts.sort_values(by=["count"], ascending=False)
    selected_containers = container_counts.head(10)["container"]

    cpu_df = cpu_df[cpu_df["container"].isin(selected_containers.values)]

    grouped = cpu_df.groupby("container")

    data2plot = []
    names = []
    for name, group in grouped:
        print(name)
        names.append(name)
        df = pd.DataFrame({"value":group["cpu"]})
        X_test = create_timeseries_features_round2(df)
        #y_predict = np.round(clf_loaded.predict(X_test))
        y_predict = clf_loaded.predict(X_test)
        y_predict = np.array([1 if v > 0.95 else 0 for v in y_predict])
        data2plot.append((df["value"], y_predict))
        names.append("label")
        data2plot.append((y_predict, None))
        #y_predict = clf_loaded.predict(X_test)

    plot_labled_data(data2plot, names, "cpu_anomalies.png")




feature_based_forecast()
#explore_features_in_ts()

#find_best_cutoff_with_neighbourhood_adjustment()

#kfold_forecast()

#print(0.99**100)

#check_cpu_data()




