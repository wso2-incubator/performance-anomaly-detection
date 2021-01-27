# Read all files in smd_input/ServerMachineDataset/test, standardize them along columns, merge them together with labels, rename columns
import csv
from os import listdir
from os.path import isfile, join
import numpy as np

import pandas as pd

PERCENTILE_VALUE = 99
# Change False to True as necessary
CONVERT_TEST_DATA_TO_CSV = False
CONVERT_TEST_LABELS_TO_CSV = False
MERGE_TEST_DATA_AND_LABELS = False
NORMALISE_TEST_DATA = False
# PREPARE_SMD_ATTACK = True
MERGE_NORMALISED_TEST_DATA_AND_LABELS = True

test_dir_path = 'test/'
output_dir_path = '../smd/test/'
test_label_dir_path = 'test_label/'
label_output_dir_path = '../smd/test_label/'
normalised_test_dir_path = '../smd/normalised_test/'

# List all files in ServerMachineDataset/test directory and convert those into csv files. Store them in ServerMachineDataset/temp_csvs/test
machine_file_list = [f for f in listdir(test_dir_path) if isfile(join(test_dir_path, f))]
if CONVERT_TEST_DATA_TO_CSV:
    for machine_file in machine_file_list:
        with open(test_dir_path + machine_file, 'r') as in_file:
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

# List all files in ServerMachineDataset/test_label directory and convert those into csv files. Store them in ServerMachineDataset/temp_csvs/test_label
machine_file_list = [f for f in listdir(test_label_dir_path) if isfile(join(test_label_dir_path, f))]
if CONVERT_TEST_LABELS_TO_CSV:
    for machine_file in machine_file_list:
        with open(test_label_dir_path + machine_file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            output_file = label_output_dir_path + machine_file.strip('.txt') + ".csv"
            column_names = ['Normal/Attack']
            with open(output_file, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(column_names)
                writer.writerows(lines)

machine_file_list = [f for f in listdir(output_dir_path) if isfile(join(output_dir_path, f))]
if MERGE_TEST_DATA_AND_LABELS:
    data_sets = []
    for machine_file in machine_file_list:
        df = pd.read_csv(output_dir_path+machine_file)
        label_df = pd.read_csv(label_output_dir_path+machine_file)
        new_df = pd.concat([df, label_df], axis=1)
        data_sets.append(new_df)
    data_set = pd.concat(data_sets).reset_index(drop=True)
    data_set.to_csv('../smd/SMD_Dataset_Attack_non_normalised.csv', index=True)  # to drop index while writing to csv, set index to False

# No need to standardize across columns ATM - Remove the below code
# if PREPARE_SMD_ATTACK:
#     df = pd.read_csv('../smd/SMD_Dataset_Attack_non_normalised.csv')
#     df = df.drop(["Unnamed: 0"], axis=1)
#     for column_heading in df.columns.values:
#         if not (column_heading == 'Normal/Attack'):
#             max_val = np.percentile(df[column_heading], [PERCENTILE_VALUE])[0]
#             df[column_heading] = (df[column_heading] / max_val).fillna(0)
#         else:
#             df[column_heading] = df[column_heading]
#     df.to_csv('../smd/SMD_Dataset_Attack.csv', index=True) # to drop index while writing to csv, set index to False

machine_file_list = [f for f in listdir(output_dir_path) if isfile(join(output_dir_path, f))]
if NORMALISE_TEST_DATA:
    # For each application standardize across columns and merge all normalised apps together
    print (machine_file_list)
    for machine_file in machine_file_list:
        print(machine_file)
        df = pd.read_csv(output_dir_path+machine_file)
        print(df)
        for column_heading in df.columns.values:
            max_val = np.percentile(df[column_heading], [PERCENTILE_VALUE])[0]
            df[column_heading] = (df[column_heading] / max_val).fillna(0)
        print(df)
        print("--------------------")
        df.to_csv(normalised_test_dir_path+machine_file, index=True)  # to drop index while writing to csv, set index to False

machine_file_list = [f for f in listdir(normalised_test_dir_path) if isfile(join(normalised_test_dir_path, f))]
if MERGE_NORMALISED_TEST_DATA_AND_LABELS:
    data_sets = []
    for machine_file in machine_file_list:
        df = pd.read_csv(normalised_test_dir_path+machine_file)
        label_df = pd.read_csv(label_output_dir_path+machine_file)
        new_df = pd.concat([df, label_df], axis=1)
        data_sets.append(new_df)
    data_set = pd.concat(data_sets).reset_index(drop=True)
    data_set.to_csv('../smd/SMD_Dataset_Attack.csv', index=False)  # to drop index while writing to csv, set index to False
