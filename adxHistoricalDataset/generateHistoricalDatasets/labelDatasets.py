import os
import pandas as pd
# Read csv files in multivariateDatasets folder
# Add a new column 'is_anomaly' with all values 0
# Save file by same name in labeledDatasets folder
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experimentFolder', 'multivariateDatasets'))
dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experimentFolder', 'labeledDatasets'))
list_of_files = [filename for filename in os.listdir(dir) if filename.endswith(".csv")]
for file in list_of_files:
    df = pd.read_csv(dir + "/" + file)
    labels_list=[0]*len(df)
    is_anomaly = pd.DataFrame({'is_anomaly': labels_list})
    df_new = df.merge(is_anomaly, left_index = True, right_index = True)
    df_new.to_csv(dest_dir+"/"+file, index=False)
