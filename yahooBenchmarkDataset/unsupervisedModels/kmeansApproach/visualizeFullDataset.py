# imports
from yahooBenchmarkDataset.unsupervisedModels.kmeansApproach import datasetClassifier
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# tags
# 0 - spikes only, 1 - level shifts only, 2 - dips only, 3 - mixed anomalies, 4 - other types of anomalies (e.g. violation/change of seasonality), 5 - no anomalies
CATEGORY_LABELS = {0:'spikes_only',1:'level_shifts_only',2:'dips_only',3:'mixed_anomaly',4:'other_anomaly',5:'no_anomaly',6:'all_datasets_combined'}
SIDE_BY_SIDE_VISUALIZATION_CATEGORICAL_DATASETS = {'Required':True, 'Categories':[0,1,2,3,4,5,6]}

def add_separator_column(df):
    df["separator_column"] = [0]*len(df)
    df.iloc[-1, df.columns.get_loc('separator_column')] = 1
    return df

def normalize_dataset(df):
    # Normalize the value column.
    max_val = np.percentile(df["value"], [99])[0]
    df["value"] = df["value"] / max_val
    return df

def load_dataset(category):
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','ydata-labeled-time-series-anomalies-v1_0','A1Benchmark'))
    switcher = {
        0 : datasetClassifier.Classifier.spikes_only,
        1 : datasetClassifier.Classifier.level_shifts_only,
        2 : datasetClassifier.Classifier.dips_only,
        3 : datasetClassifier.Classifier.mixed_anomaly,
        4 : datasetClassifier.Classifier.other_anomaly,
        5 : datasetClassifier.Classifier.no_anomaly,
        6 : datasetClassifier.Classifier.all_datasets_combined
    }
    list_of_files = ["/real_"+str(f)+".csv" for f in switcher[category]]
    data_sets = [pd.read_csv(dir+i) for i in list_of_files]
    separated_datasets = [add_separator_column(df) for df in data_sets]
    transformed_datasets = [normalize_dataset(df) for df in separated_datasets]
    data_set = pd.concat(transformed_datasets)
    data_set = data_set.fillna(0)
    data_set = data_set.drop(["timestamp"], axis=1)
    return data_set

# Plot all datasets side by side with anomaly points plotted in red
def plot_dataset_sidebyside(required_categories):
    for category in required_categories:
        df = load_dataset(category)
        df["Index"] = range(df.shape[0])
        df["is_anomaly_labels"] = [df["value"].values[i] if df["is_anomaly"].values[i] == 1 else 0 for i in
                                         range(df.shape[0])]
        df["is_separator"] = [df["value"].values[i] if df["separator_column"].values[i] == 1 else 0 for i in
                                         range(df.shape[0])]
        plt.figure(figsize=(100, 10))
        ax = sns.lineplot(x="Index", y="value", label="Values", data=df)
        ax = sns.scatterplot(x="Index", y="is_anomaly_labels", data=df, ax=ax, label="Anomaly points", color='r', s=100)
        ax = sns.scatterplot(x="Index", y="is_separator", data=df, ax=ax, label="Separation points", color='k', s=500)

        ax.legend()
        plt.savefig("visualizationFigures/fullDatasetVisualizationDirectory/"+CATEGORY_LABELS[category]+"_full_data_set.png")

def main():
    if SIDE_BY_SIDE_VISUALIZATION_CATEGORICAL_DATASETS['Required']:
        # This will save full dataset visualization figures in visualizationFigures/fullDatasetVisualizationDirectory/
        plot_dataset_sidebyside(SIDE_BY_SIDE_VISUALIZATION_CATEGORICAL_DATASETS['Categories'])

main()