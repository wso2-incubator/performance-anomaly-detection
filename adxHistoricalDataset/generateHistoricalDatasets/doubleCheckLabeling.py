import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_for_evaluation(file, X_test):
    X_test["Index"] = range(X_test.shape[0])
    X_test["anom"] = [X_test["throughput"].values[i] if X_test["is_anomaly"].values[i] == 1 else -1 for i in
                               range(X_test.shape[0])]
    plt.figure(figsize=(100, 10))
    ax = sns.lineplot(x="Index", y="throughput", data=X_test)
    ax = sns.scatterplot(x="Index", y="anom", data=X_test, ax=ax, label="Anomaly", color='r', s=500)
    ax.legend()
    plt.savefig("visualizeData/manualLabeling/double_check/" + file + ".png")

# Read labeled csv files from experimentFolder/labeledDatasets
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experimentFolder', 'labeledDatasets'))
list_of_files = [filename for filename in os.listdir(source_dir) if filename.endswith(".csv")]
print(list_of_files)
# Plot them with anomalies marked in red
for file in list_of_files:
    df = pd.read_csv(source_dir + "/" + file)
    file_name = file.strip('.csv')
    plot_for_evaluation(file_name, df)


