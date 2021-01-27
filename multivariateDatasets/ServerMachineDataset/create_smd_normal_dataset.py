# Read all files in ServerMachineDataset/train, standardize them along columns, merge them, rename columns
import csv
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

PERCENTILE_VALUE = 99
# Change False to True as necessary
CONVERT_TRAIN_DATA_TO_CSV = False
PREPARE_SMD_NORMAL = True
MERGE_TRAIN_DATA = False

train_dir_path = 'train/'
output_dir_path = '../smd/train/'

# List all files in ServerMachineDataset/train directory and convert those into csv files. Store them in ServerMachineDataset/temp_csvs/train
machine_file_list = [f for f in listdir(train_dir_path) if isfile(join(train_dir_path, f))]
if CONVERT_TRAIN_DATA_TO_CSV:
    for machine_file in machine_file_list:
        with open(train_dir_path + machine_file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            output_file = output_dir_path + machine_file.strip('.txt') + ".csv"
            column_names = []
            for i in range(1,39):
                column_names.append("col"+str(i))
            with open(output_file, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(column_names)
                writer.writerows(lines)

if PREPARE_SMD_NORMAL:
    # List all files in SMD/train directory
    csv_file_list = [f for f in listdir(output_dir_path) if isfile(join(output_dir_path, f))]
    data_sets = []
    for csv_file in csv_file_list:
        df = pd.read_csv(output_dir_path+csv_file)
        for column_heading in df.columns.values:
            max_val = np.percentile(df[column_heading], [PERCENTILE_VALUE])[0]
            df[column_heading] = (df[column_heading] / max_val).fillna(0)
        data_sets.append(df)
        print(df.shape)
    data_set = pd.concat(data_sets).reset_index(drop=True)
    data_set.to_csv('../smd/SMD_Dataset_Normal.csv', index=True) # to drop index while writing to csv, set index to False

machine_file_list = [f for f in listdir(output_dir_path) if isfile(join(output_dir_path, f))]
if MERGE_TRAIN_DATA:
    data_sets = []
    for machine_file in machine_file_list:
        df = pd.read_csv(output_dir_path+machine_file)
        data_sets.append(df)
    data_set = pd.concat(data_sets).reset_index(drop=True)
    data_set.to_csv('../smd/SMD_Dataset_Normal_non_normalised.csv', index=True)  # to drop index while writing to csv, set index to False