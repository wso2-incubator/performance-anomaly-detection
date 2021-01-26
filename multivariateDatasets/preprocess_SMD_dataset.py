import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

PREPARE_METADATA_FILES = False
PROCESS_MATADATA_FILES = True

if PREPARE_METADATA_FILES:
    # Read the SMD train and test sets and print the ranges
    # Arrange the value ranges in a dataframe with machines as rows and columns of each dataset as columns
    train_dir_path = 'smd/train/'
    test_dir_path = 'smd/test/'

    train_machine_file_list = [f for f in listdir(train_dir_path) if isfile(join(train_dir_path, f))]
    test_machine_file_list = [f for f in listdir(test_dir_path) if isfile(join(test_dir_path, f))]

    # Initialize the train dataset creation dictionary
    train_df = pd.read_csv(str(train_dir_path)+train_machine_file_list[0])
    train_columns = train_df.columns.values
    train_data = {k: [] for k in train_columns}

    for file_name in train_machine_file_list:
        # print(file_name)
        df = pd.read_csv(str(train_dir_path)+file_name)
        columns = df.columns.values
        for column in columns:
            max_value = round(df[column].max(),2)
            min_value = round(df[column].min(),2)
            # print(str(min_value)+"-"+str(max_value))
            train_data[column].append(str(min_value)+"-"+str(max_value))

    # print(train_data)
    train_details_df = pd.DataFrame(data=train_data)
    print(train_details_df)
    # Save the dataframe to a csv
    train_details_df.to_csv('smd_train_metadata.csv')

    test_df = pd.read_csv(str(test_dir_path)+test_machine_file_list[0])
    test_columns = test_df.columns.values
    test_data = {k: [] for k in test_columns}

    for file_name in test_machine_file_list:
        # print(file_name)
        df = pd.read_csv(str(test_dir_path)+file_name)
        columns = df.columns.values
        for column in columns:
            max_value = round(df[column].max(),2)
            min_value = round(df[column].min(),2)
            # print(str(min_value)+"-"+str(max_value))
            test_data[column].append(str(min_value) + "-" + str(max_value))

    # print(test_data)
    test_details_df = pd.DataFrame(data=test_data)
    print(test_details_df)
    # Save the dataframe to a csv
    test_details_df.to_csv('smd_test_metadata.csv')

if PROCESS_MATADATA_FILES:
    # Create a dataframe containing trainset and testset as rows and columns of each dataset as columns
    train_df=pd.read_csv('smd_train_metadata.csv')
    columns=train_df.columns.values
    new_columns = np.delete(columns, [0])
    final_dictionary = {k: [] for k in new_columns}
    for column in columns:
        if column!='Unnamed: 0':
            min_val_across_col = float(train_df[column][0].split('-')[0])
            max_val_across_col = float(train_df[column][0].split('-')[1])
            print("-------initial min max values for " + str(column) + "-------")
            # print(min_val_across_col)
            # print(max_val_across_col)
            # print("--------------")
            for line in (train_df[column][1:]):
                min_val = float(line.split('-')[0])
                max_val = float(line.split('-')[1])
                # print(str(min_val) + str(min_val < min_val_across_col))
                if min_val < min_val_across_col:
                    min_val_across_col = min_val
                    # print("New min_val_across_col is " + str(min_val_across_col))
                # print(str(max_val) + str(max_val > max_val_across_col))
                if max_val > max_val_across_col:
                    max_val_across_col = max_val
                    # print("New max_val_across_col is " + str(max_val_across_col))
            print(str(min_val_across_col)+"-"+str(max_val_across_col))
            final_dictionary[column].append(str(min_val_across_col)+"-"+str(max_val_across_col))

    print("------------------------")
    test_df = pd.read_csv('smd_test_metadata.csv')
    columns = test_df.columns.values
    for column in columns:
        if column != 'Unnamed: 0':
            min_val_across_col = float(test_df[column][0].split('-')[0])
            max_val_across_col = float(test_df[column][0].split('-')[1])
            print("-------initial min max values for "+str(column)+"-------")
            # print(min_val_across_col)
            # print(max_val_across_col)
            # print("--------------")
            for line in (test_df[column][1:]):
                min_val = float(line.split('-')[0])
                max_val = float(line.split('-')[1])
                # print(str(min_val)+str(min_val < min_val_across_col))
                if min_val < min_val_across_col:
                    min_val_across_col = min_val
                    # print("New min_val_across_col is " + str(min_val_across_col))
                # print(str(max_val) + str(max_val > max_val_across_col))
                if max_val > max_val_across_col:
                    max_val_across_col = max_val
                    # print("New max_val_across_col is " + str(max_val_across_col))
            print(str(min_val_across_col) + "-" + str(max_val_across_col))
            final_dictionary[column].append(str(min_val_across_col) + "-" + str(max_val_across_col))
    metadata_df = pd.DataFrame(data=final_dictionary)
    print(metadata_df)
    # Save the dataframe to a csv
    metadata_df.to_csv('smd_metadata.csv')