from wso2.anomaly_detection import *

def build_yahoo_dataset():
    dir = "/Users/srinath/playground/Datasets/SystemsData/YahooAnomalyDataset/A1Benchmark/"
    file_index_list = [11, 19, 15, 9, 45, 56, 59, 64, 41, 16, 54, 39, 2, 44, 23, 18, 47, 31, 24, 49, 26, 3, 0, 60, 52, 40, 43,
                 61, 12, 17, 33, 20, 1, 38, 7, 50, 30, 32, 21, 48, 5, 35, 4, 14, 29, 22, 55, 8, 62, 6, 58, 36, 53, 34,
                 10, 63, 28, 51, 25, 42, 57, 46, 13, 37, 27]

    file_list = [dir+"real_" +str(fi+1)+".csv" for fi in file_index_list]
    detector = build_univairiate_model(file_list=file_list, value_feature_name="value", test_fraction=0.3)

    file_name = "yahoo_model.sav"
    detector.save_model(file_name)

    new_detector = UnivariateAnomalyDetector(file_name)

build_yahoo_dataset()
