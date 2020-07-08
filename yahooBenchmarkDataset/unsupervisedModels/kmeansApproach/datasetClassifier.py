class Classifier:
    spikes_only = [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,21,30,33,41,42,45,50,60,62,65,66,67]
    level_shifts_only = [7, 17, 19, 22, 25, 48, 58, 63]
    dips_only = [18,23,29,36,38,49,51,52,55,56]
    mixed_anomaly = [20,24,27,28,31,32,34,37,39,40,43,44,46,53,54,61]
    other_anomaly = [26,47,57]
    no_anomaly = [35,59,64]
    all_datasets_combined = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                             26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                             49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

class FoldsClassifier:
    all_datasets_combined = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                             26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                             49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    # category_labels contains the labels for each type of data set based on the anomalies they carry
    # 0 - spikes only, 1 - level shifts only, 2 - dips only, 3 - mixed anomalies, 4 - other types of anomalies (e.g. violation/change of seasonality), 5 - no anomalies
    category_labels = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 0, 1, 2, 3, 1, 4, 3, 3, 2, 0, 3, 3, 0,
                       3, 5, 2, 3, 2, 3, 3, 0, 0, 3, 3, 0, 3, 4, 1, 2, 0, 2, 2, 3, 3, 2, 2, 4, 1, 5, 0, 3, 0, 1, 5, 0, 0, 0]