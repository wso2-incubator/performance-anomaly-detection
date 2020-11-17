"""
@author: Guansong Pang (with major refactor by Jakub Karczewski)
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""
import pickle
from os.path import join
from collections import defaultdict
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc, \
    average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.backend import mean, std, abs, maximum
from tensorflow.keras.regularizers import l2

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_INT = np.iinfo(np.int32).max

ANOMALY_LABEL_COLUMN_NAME = "is_anomaly"
INDEX_COLUMN = "Unnamed: 0"
GRU_or_LSTM = "LSTM"

def lstm_function(input):
    # Add an LSTM layer
    print("input is of shape " + str(input.shape) + " and type " + str(type(input)))
    # print(input)
    lstm = tf.keras.layers.LSTM(256)
    lstm_output = lstm(input)
    return lstm_output

def gru_function(input):
    # Add an GRU layer
    print("input is of shape " + str(input.shape) + " and type " + str(type(input)))
    # print(input)
    gru = tf.keras.layers.GRU(128)
    gru_output = gru(input)
    return gru_output

class DevNet:
    """Deviation Network for credit card fraud detection."""
    def __init__(self, epochs, batch_size, num_runs, seed, output_path='output'):
        self.output_path = output_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_runs = num_runs

        self.seed = seed

        self.scaler = MinMaxScaler()
        self.random_state = np.random.RandomState(self.seed)
        # 95% confidence threshold score value
        self.conf_threshold = 1.96

    @staticmethod
    @tf.function
    def deviation_loss(y_true, y_pred):
        """Z-score based deviation loss"""
        confidence_margin = 5.
        ref = tf.cast(np.random.normal(loc=0., scale=1.0, size=5000), dtype=tf.float32)
        dev = (y_pred - mean(ref)) / std(ref)
        normal_loss = abs(dev)
        anomaly_loss = abs(maximum(confidence_margin - dev, 0.))
        return mean((1 - y_true) * normal_loss + y_true * anomaly_loss)

    def deviation_network(self, input_shape, l2_coef=0.01):
        """Construct the deviation network-based detection model."""
        x_input = Input(shape=input_shape)
        # dev_network_s
        # intermediate = Dense(20, activation='relu', kernel_regularizer=l2(l2_coef), name='hl1')(x_input)
        # intermediate = Dense(1, activation='linear', name='score')(intermediate)
        # dev_network_d
        intermediate = Dense(1000, activation='relu',
                             kernel_regularizer=l2(l2_coef), name='hl1')(x_input)
        intermediate = Dense(250, activation='relu',
                             kernel_regularizer=l2(l2_coef), name='hl2')(intermediate)
        intermediate = Dense(20, activation='relu',
                             kernel_regularizer=l2(l2_coef), name='hl3')(intermediate)
        intermediate = Dense(1, activation='linear', name='score')(intermediate)
        model = Model(x_input, intermediate)
        rms = RMSprop(clipnorm=1.)
        model.compile(loss=self.deviation_loss, optimizer=rms)
        return model

    def get_data_generator(self, x, y):
        """Generates batches of training data."""
        # Determine anomaly indices and normal indices
        anomaly_indexes = np.where(y == 1)[0]
        normal_indexes = np.where(y == 0)[0]
        while True:
            ref, training_labels = self.get_batch(x, anomaly_indexes, normal_indexes)
            yield ref.astype('float32'), training_labels.astype('float32') # casting to float32

    def get_batch(self, x_train, anomaly_indexes, normal_indexes):
        """Preprocess training set by alternating between negative and positive pairs."""
        preprocessed_x_train = np.empty((self.batch_size, x_train.shape[-1]))
        training_labels = []
        n_normal = len(normal_indexes)
        n_anomaly = len(anomaly_indexes)
        for i in range(len(preprocessed_x_train)):
            if i % 2 == 0:
                selected_idx = self.random_state.choice(n_normal, 1)
                preprocessed_x_train[i] = x_train[normal_indexes[selected_idx]]
                training_labels += [0]
            else:
                selected_idx = self.random_state.choice(n_anomaly, 1)
                preprocessed_x_train[i] = x_train[anomaly_indexes[selected_idx]]
                training_labels += [1]
        return np.array(preprocessed_x_train), np.array(training_labels) # returns a batch containing both normal and anomaly points

    @staticmethod
    def round_top_perc(preds, perc):
        return np.where(preds > np.percentile(preds, perc, interpolation='nearest'), 1, 0)

    def get_metrics(self, gt, preds, convert_to_proba=True): #gt is the y_test (actual labels)
        """Returns performance metrics."""
        pred_3_perc = self.round_top_perc(np.abs(preds), 97)
        pred_1_perc = self.round_top_perc(np.abs(preds), 99)
        # predictions are thresholded only if convert_to_proba=True
        if convert_to_proba:
            preds = np.array([1 if abs(x) >= self.conf_threshold else 0 for x in preds])
        roc_auc = roc_auc_score(gt, preds) # calculate roc_auc for actual_labels and predictions
        _, rec_1, _, _ = precision_recall_fscore_support(gt, pred_1_perc)
        _, rec_3, _, _ = precision_recall_fscore_support(gt, pred_3_perc)
        precision, recall, _ = precision_recall_curve(gt, preds)
        prec_rec_auc = auc(recall, precision)
        avg_precision = average_precision_score(gt, preds)
        return {
            'roc_auc': np.around(roc_auc, decimals=4),
            'prec_rec_auc': np.around(prec_rec_auc, decimals=4),
            'recall_1%': np.around(rec_1[-1], decimals=5),
            'recall_3%': np.around(rec_3[-1], decimals=5),
            'avg_precision': np.around(avg_precision, decimals=5)
        }

    @staticmethod
    def plot_loss(history):
        """Plot training & validation loss values."""
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def run_devnet(self, lstm_array_for_x_reshaped, y):
        # create placeholder variables for scores in each run
        all_metrics = dict()
        for metrics_type in ('authors', 'mine'):
            all_metrics[metrics_type] = {name: np.zeros(self.num_runs) for name in
                                         ('roc_auc', 'prec_rec_auc', 'recall_1%', 'recall_3%', 'avg_precision')}
        print(f'lstm_array_for_x_reshaped shape {lstm_array_for_x_reshaped.shape}')
        # inspect number of frauds (anomalies)
        anomalies = np.count_nonzero(y == 1)
        print(f'There are {anomalies} original frauds (outliers) in the dataset.')
        # run training several times
        for run_id in np.arange(self.num_runs):
            # split data into train/test
            x_train, x_test, y_train, y_test = train_test_split(lstm_array_for_x_reshaped, y, test_size=0.2, random_state=self.seed, stratify=y)
            print(f'Run number: {run_id}')
            original_train_anomaly_indices = np.where(y_train == 1)[0] # Identify anomaly labels in training set
            num_train_anomalies = len(original_train_anomaly_indices)
            print(f"Original training size for run {run_id}: {x_train.shape[0]}, No. outliers: {num_train_anomalies}")
            # Note that the code here to preprocess was temporarily removed
            print(f"Post-transformations training size for run {run_id}: {x_train.shape[0]},"
                  f" No. outliers: {len(np.where(y_train == 1)[0])}")

            input_shape = x_train.shape[1:]
            print("input_shape is "+str(input_shape))
            # create model
            model = self.deviation_network(input_shape)
            model_name = f'devnet_run_{run_id}.h5'
            callbacks = [
                ModelCheckpoint(join(self.output_path, model_name), monitor='loss', verbose=0, save_best_only=True),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=0, mode='auto', baseline=None,
                              restore_best_weights=True)
            ]
            # Split the train set again for training and testing
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=self.seed,
                                                              stratify=y_train)
            train_generator = self.get_data_generator(x_train, y_train)
            val_generator = self.get_data_generator(x_val, y_val)

            # training the model for the given number of epochs; 150
            history = model.fit(train_generator, validation_data=val_generator,  epochs=self.epochs,
                                steps_per_epoch=len(x_train)//self.batch_size,
                                validation_steps=len(x_val)//self.batch_size, callbacks=callbacks, verbose=0)
            # predict for x_test
            preds = model.predict(x_test)

            # after this step, results for the run will be updated in all_metrics dictionary
            for metrics_type, fix in zip(all_metrics, (False, True)):
                for metric_name, value in self.get_metrics(y_test, preds, convert_to_proba=fix).items():
                    all_metrics[metrics_type][metric_name][run_id] = value

            with open(join(self.output_path, f'history_run:{run_id}.pkl'), 'wb') as f:
                pickle.dump(history.history, f)

        metrics_report = dict()
        for metrics_type in all_metrics:
            partial_metrics_report = defaultdict(lambda: {'avg': None, 'std': None})
            for metric_label, values in all_metrics[metrics_type].items():
                print(f'For {metrics_type} metric: {metric_label} -> avg: {values.mean()}, std: {values.std()}')
                partial_metrics_report[metric_label]['avg'] = values.mean()
                partial_metrics_report[metric_label]['std'] = values.std()
            metrics_report[metrics_type] = partial_metrics_report

        with open(join(self.output_path, 'metrics.json'), 'w') as f:
            json.dump(dict(metrics_report), f)

        return {
            'model': model,
            'predictions': preds,
            'ground_truth': y_test,
            'history': history.history,
            'metrics': metrics_report
        }

    def eval_devnet(self, model, lstm_array_for_x_reshaped):
        print(f'lstm_array_for_x_reshaped shape {lstm_array_for_x_reshaped.shape}')
        # predict for x_test
        preds = model.predict(lstm_array_for_x_reshaped)
        # 95th percentile of preds
        self.conf_threshold = np.percentile(preds, 99)
        print(self.conf_threshold)
        predictions = np.array([1 if abs(x) >= self.conf_threshold else 0 for x in preds])
        return predictions

class Evaluation:
    def __init__(self, y_test, predict_test):
        self.accuracy = accuracy_score(y_test, predict_test)
        self.precision = precision_score(y_test, predict_test)
        self.recall = recall_score(y_test, predict_test)
        self.auc = roc_auc_score(y_test, predict_test)
        self.f1_score = f1_score(y_test, predict_test)
        self.cm = confusion_matrix(y_test, predict_test)

    def print(self):
        print("Accuracy\tPrecision\tRecall\tAUC\tF1")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (self.accuracy, self.precision, self.recall, self.auc, self.f1_score))

        print("Confusion Matrix")
        print(self.cm)

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
        labels.append(target_data[i + target_steps -2])
    return np.array(data), np.array(labels)

if __name__ == '__main__':
    # Test for GPU
    is_gpu = tf.test.is_gpu_available()
    print(f"GPU is{'' if is_gpu else ' not'} available.")

    dev_net_conf = {
        'batch_size': 512,
        'num_runs': 10,
        'seed': SEED,
        'epochs': 150
    }
    # Create the lstm layers before training and testing. Then, the lstm layers will be prepared for the dataset only once.
    # read data
    dataset_path = 'dataset/yws5_for_lstm.csv'
    dataset = pd.read_csv(dataset_path)
    # Perform scaling if required
    y = dataset[ANOMALY_LABEL_COLUMN_NAME].values  # y contains anomaly labels
    x = dataset.drop(columns=[ANOMALY_LABEL_COLUMN_NAME, INDEX_COLUMN]).values
    start_index = 0
    target_steps = 1
    end_index = len(y) + 1
    history_steps = 150
    x_reshaped, y = create_rnn_univariate_data_with_targetdata(x, y, start_index, end_index, history_steps, target_steps)
    if GRU_or_LSTM == "LSTM":
        output_for_x_reshaped = lstm_function(x_reshaped)
        array_for_x_reshaped = np.array(output_for_x_reshaped)
    else:
        output_for_x_reshaped = gru_function(x_reshaped)
        array_for_x_reshaped = np.array(output_for_x_reshaped)
    # Split training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(array_for_x_reshaped, y, test_size=0.2,
                                                        random_state=SEED, stratify=y)
    # Send x_train (with y_train) as training dataset and x_test (with y_test) as testing dataset
    dev_net = DevNet(**dev_net_conf)
    results = dev_net.run_devnet(x_train,y_train)
    # Temporarily comment plotting
    # DevNet.plot_loss(results['history'])
    # Perform prediction for the entire dataset on the model and evaluate
    predictions = dev_net.eval_devnet(results['model'], x_test)
    evaluator = Evaluation(y_test, predictions)
    evaluator.print()
    # Adjust predictions for neighbourhood.
    print("Adjusted predictions for neighbourhood")
    adjusted_predictions = adjust_predictions_for_neighbourhood(y_test,predictions)
    adj_evaluator = Evaluation(y_test, adjusted_predictions)
    adj_evaluator.print()