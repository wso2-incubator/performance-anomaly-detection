import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch.utils.data as data_utils
import torch
from usad import *
from sklearn import metrics

class Evaluation:
    def __init__(self, y_test, predict_test):
        self.accuracy = metrics.accuracy_score(y_test, predict_test)
        self.precision = metrics.precision_score(y_test, predict_test)
        self.recall = metrics.recall_score(y_test, predict_test)
        self.auc = metrics.roc_auc_score(y_test, predict_test)
        self.f1_score = metrics.f1_score(y_test, predict_test)
        self.cm = metrics.confusion_matrix(y_test, predict_test)

    def print(self):
        print("Accuracy\tPrecision\tRecall\tAUC\tF1")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (self.accuracy, self.precision, self.recall, self.auc, self.f1_score))

        print("Confusion Matrix")
        print(self.cm)

    def obtain_vals(self):
        return (self.accuracy, self.precision, self.recall, self.auc, self.f1_score)

def adjust_predictions_for_neighbourhood(y_test, predict_test, slack=5):
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

if __name__ == '__main__':
    #Read data
    normal = pd.read_csv("../smd/SMD_Dataset_Normal.csv")#, nrows=1000)
    print(normal.columns.values)
    normal = normal.drop(['Unnamed: 0'] , axis = 1)
    print(normal.shape)

    # Transform all columns into float64
    for i in list(normal):
        normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
    normal = normal.astype(float)

    # min_max_scaler = preprocessing.MinMaxScaler()
    #
    # x = normal.values
    # x_scaled = min_max_scaler.fit_transform(x)
    # normal = pd.DataFrame(x_scaled)
    print(normal.head(2))

    #Read data
    attack = pd.read_csv("../smd/SMD_Dataset_Attack.csv")#, sep=";")#, nrows=1000)
    y = attack.pop("Normal/Attack")
    attack = attack.drop(['Unnamed: 0'], axis=1)
    print(attack.shape)

    # Transform all columns into float64
    for i in list(attack):
        attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
    attack = attack.astype(float)

    # x = attack.values
    # x_scaled = min_max_scaler.transform(x)
    # attack = pd.DataFrame(x_scaled)
    print(attack.head(2))

    window_size=5 # originally 12
    windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
    print(attack.shape)
    windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
    # print("*************")
    # print(np.arange(window_size)[None, :])
    # print(np.arange(attack.shape[0]-window_size)[:, None])
    # print("*************")
    print(windows_attack.shape)
    BATCH_SIZE = 7919
    N_EPOCHS = 250 # originally 100
    hidden_size = 1 # originally 10

    w_size=windows_normal.shape[1]*windows_normal.shape[2]
    print(w_size)
    z_size=windows_normal.shape[1]*hidden_size

    windows_normal_train = windows_normal[:int(np.floor(.8 * .5 * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor(.8 * .5 * windows_normal.shape[0])):int(np.floor(.5 * windows_normal.shape[0]))]
    windows_normal_test = windows_normal[int(np.floor(.5 * windows_normal.shape[0])):]

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(np.concatenate([windows_normal_test,windows_attack])).float().view(([windows_normal_test.shape[0]+windows_attack.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UsadModel(w_size, z_size)
    model = to_device(model,device)

    history = training(N_EPOCHS,model,train_loader,val_loader)

    # Turn off if not needed
    # plot_history(history)

    torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder1': model.decoder1.state_dict(),
                'decoder2': model.decoder2.state_dict()
                }, "model.pth")

    checkpoint = torch.load("model.pth")

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])

    results=testing(model,test_loader)
    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                  results[-1].flatten().detach().cpu().numpy()])
    # y_fake_test is how the authors of the paper originally calculate the test labels. actual_labels_np_array
    #   introduced later would be the correct labels list
    y_fake_test=np.concatenate([np.zeros(windows_normal_test.shape[0]),
                           np.ones(windows_attack.shape[0])])
    actual_labels = []
    # Fill actual_labels with anomaly labels
    # For y values from [window_size-1] to [len(y)-1], create actual_labels
    # for element in y[window_size-1:len(y)-1]:
    #     if element=='Attack' or element=='A ttack':
    #         actual_labels.append(1)
    #     elif element=='Normal':
    #         actual_labels.append(0)
    # However, in the paper, if at least one point in the window is an anomaly the whole window is anomalous.
    y_list = y.tolist()
    for i in range(window_size-1,len(y_list)-1):
        if 1 in y_list[i-window_size+1:i+1]:
            actual_labels.append(1)
        else:
            actual_labels.append(0)
    actual_labels_np_array=np.asarray(actual_labels)
    y_test=np.concatenate([np.zeros(windows_normal_test.shape[0]),actual_labels_np_array])
    # Turn off plotting the results histogram if not needed
    # histogram(y_test,y_pred)
    # threshold=ROC(y_test,y_pred)
    # Get 95th percentile of y_pred values
    threshold=np.percentile(y_pred, [99])[0] # 95th percentile is considered in swat case
    # Turn off visualizing the confusion matrix if not needed
    # confusion_matrix(y_test,np.where(y_pred > threshold, 1, 0),perc=True)

    y_pred_for_eval = []

    for val in y_pred:
        if val > threshold:
            y_pred_for_eval.append(1)
        else:
            y_pred_for_eval.append(0)

    print("Original evaluation results")
    y_pred_for_arr = np.array(y_pred_for_eval)
    evaluator = Evaluation(y_test, y_pred_for_arr)
    evaluator.print()

    # It would not be meaningful to evaluate after adjusting neighbourhood since that is done during label formation
    # print("Neighbourhood adjusted evaluation results")
    # adjusted_predictions = adjust_predictions_for_neighbourhood(y_test, y_pred_for_arr)
    # adj_evaluator = Evaluation(y_test, adjusted_predictions)
    # adj_evaluator.print()

# Original evaluation results
# Accuracy	Precision	Recall	AUC	F1
# 0.97	0.26	0.09	0.54	0.14
# Confusion Matrix
# [[1024036    7827]
#  [  27952    2800]]
