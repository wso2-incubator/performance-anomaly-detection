import pandas as pd
import numpy as np
import scipy


def add_base_stat_features(s_train, prefix=""):
    window_list = [5, 25, 50]
    for i, w in enumerate(window_list):
        s_train[prefix + "value.w" + str(w) + ".mean"] = s_train["value"].rolling(w).mean().fillna(s_train["value"] if i ==0 else s_train[prefix + "value.w" + str(window_list[i-1]) + ".mean"])
        s_train[prefix + "value.w" + str(w) + ".std"] = s_train["value"].rolling(w).std().fillna(0 if i ==0 else s_train[prefix + "value.w" + str(window_list[i-1]) + ".std"])
        s_train[prefix + "value.w" + str(w) + ".kurt"] = s_train["value"].rolling(w).kurt().fillna(0 if i ==0 else s_train[prefix + "value.w" + str(window_list[i-1]) + ".kurt"])
        s_train[prefix + "value.w" + str(w) + ".zscore"] = s_train["value"].rolling(w).apply(lambda w: scipy.stats.zscore(w)[-1], raw=True)\
            .fillna(0 if i == 0 else s_train[prefix + "value.w" + str(window_list[i - 1]) + ".zscore"])
        s_train[prefix + "value.w" + str(w) + ".entropy"] = s_train["value"].rolling(w).apply(calculate_entropy, raw=True)\
            .fillna(0 if i == 0 else s_train[prefix + "value.w" + str(window_list[i - 1]) + ".entropy"])
    return s_train


def calculate_entropy(w):
    entropy = scipy.stats.entropy(w)
    return entropy if np.isfinite(entropy) else 10000

def mark_peaks(s_train, feild):
    '''
    Detect peaks in the give feild and add a new feature to mark thems
    :param s_train:
    :param feild:
    :return:
    '''
    s_train[feild + ".smooth"] = scipy.ndimage.gaussian_filter1d(s_train[feild].values, 10)

    peaks, _ = scipy.signal.find_peaks(s_train[feild + ".smooth"].values)
    peaks = np.sort(peaks)
    peak_indexs = np.zeros(s_train.shape[0])

    peak_count = len(peaks)
    last_peak_index = 0
    next_peak_index = 1
    print(len(peaks))
    for i in range(len(peak_indexs)):
        if i < peaks[last_peak_index]:
            peak_indexs[i] = -1
        elif i >= peaks[last_peak_index] and (next_peak_index == peak_count or i <peaks[next_peak_index]):
            peak_indexs[i] = i - peaks[last_peak_index]
        elif  i == peaks[next_peak_index]:
            peak_indexs[i] = 0
            last_peak_index = next_peak_index
            next_peak_index = next_peak_index + 1
        else:
            raise Exception("this should never happen i=", i, "last_peak_index", last_peak_index, "next_peak_index", next_peak_index)

    s_train[feild+".peaks"] = peak_indexs
    return s_train


def get_seasonal_component(w):
    return get_seasonal_window(w)[-1]


def get_seasonal_window(w):
    frequencies = np.fft.fft(w)
    # TODO I am taking largest real value, is that right
    freq_size = np.sqrt(frequencies.real * frequencies.real + frequencies.imag * frequencies.imag)

    sortedIndexs = np.argsort(freq_size)  # returns values in assending order

    index2keep = -1
    n = len(sortedIndexs)
    for i in range(n):
        if sortedIndexs[n - 1 - i] >= n / 2:
            index2keep = sortedIndexs[n - 1 - i]
            break

    period_freq = np.zeros(n, dtype=complex)
    period_freq[index2keep] = frequencies[index2keep]
    recovered_signal = np.fft.ifft(period_freq)
    return recovered_signal


def residual_without_seasonalitiy_via_acorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag - 1]
    if np.abs(r) > 0.5 and lag > 5:
        return x[-1] - x[-lag-1]
    else:
        return x[-1]


def calculate_wasserstein_distance(w):
    u = w
    v = w[-10:]
    return scipy.stats.wasserstein_distance(u, v)


def remove_sesonality_and_take_ratio(w):
    seasonal_componet = get_seasonal_component(w)
    to_devide = np.median(seasonal_componet)

    return seasonal_componet[-1]/to_devide if to_devide != 0 else 10000



def create_timeseries_features(s_train):
    '''
    1) percentage diff between two adjacent measurements (percentages based on current values)
    2) z score of a values in a moving window
    3) increasing or decreasing (and rate of increment or decrement) (1st, 2nd differentiation)

    :param s_train:
    :return:
    '''
    #max_val = np.percentile(s_train["value"], z[99])[0]
    #s_train["value"] = s_train["value"]/max_val
    #s_train["value"] = scipy.stats.zscore(s_train["value"])

    for i in [1,2,3, 5, 10, 15]:
        s_train["value.lag"+ str(i)] = s_train["value"].shift(i).fillna(0)

    s_train["first_diff"], s_train["second_diff"]  = calculate_differences(s_train["value"])

    s_train["first_long_diff"] = s_train["value.lag5"] - s_train["value.lag1"]
    s_train["second_long_diff"] = (s_train["value.lag10"] - s_train["value.lag5"]) -(s_train["value.lag5"] - s_train["value.lag1"])

    window_list = [5, 25, 50]
    for w in window_list:
        s_train["value.w."+str(w)+"mean"] = s_train["value"].rolling(w).mean()
        s_train["value.w"+str(w)+".std"] = s_train["value"].rolling(w).std()
        s_train["value.w"+str(w)+".kurt"] = s_train["value"].rolling(w).kurt()


    #Calculate ratios
    for i in [2, 3, 5, 10, 15]:
        s_train["value.lag" + str(i)+"_ratio"] = calculate_ratio(s_train["value.lag" + str(i)], s_train["value.lag1"])

    s_train["first_ratio"], s_train["second_diff_ratio"] = calculate_fs_ratios(s_train["value"])

    s_train["first_long_ratio"] = calculate_ratio(s_train["value.lag5"], s_train["value.lag1"])
    #s_train["second_long_diff"] = calculate_ratio(s_train["value.lag10"],s_train["value.lag5"]) - (
    #            s_train["value.lag5"] - s_train["value.lag1"])

    window_list = [5, 25, 50]
    for w in window_list:
        s_train["value.w." + str(w) + "mean"] = s_train["value"].rolling(w).mean()
        s_train["value.w" + str(w) + ".std"] = s_train["value"].rolling(w).std()
        s_train["value.w" + str(w) + ".kurt"] = s_train["value"].rolling(w).kurt()
        s_train["value.w" + str(w) + ".zscore"] = s_train["value"].rolling(w).apply(lambda w: scipy.stats.zscore(w)[-1])

    print(s_train[50:].head())
    print(list(s_train))

    print(s_train["is_anomaly"].value_counts())

    #s_train["first_long_diff"] = s_train["value.lag5"] - s_train["value.lag1"]
    return s_train

def calculate_differences(values):
    first_diff = [values[i] - values[i-1] if i> 1 else 0 for i in range(len(values))]
    second_diff = [first_diff[i] - first_diff[i - 1] if i > 1 else 0 for i in range(len(first_diff))]
    return first_diff, second_diff

def calculate_fs_ratios(values):
    first_diff = [values[i]/values[i-1] if i> 1 and values[i-1] > 0 else 1000 for i in range(len(values))]
    second_diff = [first_diff[i]/first_diff[i - 1] if i > 1 and first_diff[i-1]> 0 else 1000 for i in range(len(first_diff))]
    return first_diff, second_diff



def create_timeseries_features_round2(s_train):
    '''
    1) percentage diff between two adjacent measurements (percentages based on current values)
2) z score of a values in a moving window
3) increasing or decreasing (and rate of increment or decrement) (1st, 2nd differentiation)

    :param s_train:
    :return:
    '''

    '''
            Features    Importance
1      value.w50.std  24485.428284
0              value  21012.356077
12    value.w.50mean  14836.860909
3     value.w50.kurt  11325.520910
6      value.w25.std   7942.732289
10    value.w.25mean   7628.483476
2     value.w25.kurt   6296.931186
4      value.w.5mean   6100.632167
13      value.w5.std   3339.431119
18       value.lag15   3034.794136
16       value.lag10   3011.563401
9         value.lag1   2341.034196
14        value.lag5   2091.202195
11        value.lag3   1144.774942
17        value.lag2    824.416260
5    first_long_diff    608.501550
15     value.w5.kurt    524.121139
7         first_diff    434.708005
19       second_diff    220.718991
8   second_long_diff    131.213199
    '''

    #max_val = np.percentile(s_train["value"], [90])[0]
    #s_train["value"] = s_train["value"]/max_val

    #replacing with smooth hurrts F1=0.38
    #diff with smooth makes minor diff
    #s_train["value.smooth.diff"] = s_train["value"] - scipy.ndimage.gaussian_filter1d(s_train["value"].values, 10)


    s_train = add_base_stat_features(s_train)
    s_train["value.ratio"] = calculate_ratio(s_train["value"], s_train["value"].rolling(10).mean())
    s_train["value.w50.std.ratio"] = calculate_ratio(s_train["value.w50.std"], s_train["value.w50.std"].rolling(10).mean())
    s_train["value.w50.kurt.ratio"] = calculate_ratio(s_train["value.w50.kurt"], s_train["value.w50.kurt"].rolling(10).mean())
    s_train["value.w50.entropy.ratio"] = calculate_ratio(s_train["value.w50.entropy"], s_train["value.w50.entropy"].rolling(10).mean())
    s_train["value.w50.zscore.ratio"] = calculate_ratio(s_train["value.w50.zscore"],
                                                         s_train["value.w50.zscore"].rolling(10).mean())
    s_train["value.w50.mean.ratio"] = calculate_ratio(s_train["value.w50.mean"],
                                                         s_train["value.w50.mean"].rolling(10).mean())

    '''
    No clear improvement
    #wasserstein_distance between last 50 vs 10 windows(see https://www.youtube.com/watch?v=U7xdiGc7IRU&t=963s)
    s_train["wasserstein_distance.w50_10"] = s_train["value"].rolling(50).apply(calculate_wasserstein_distance, raw=True)
    s_train["wasserstein_distance.w50_10"] = s_train["wasserstein_distance.w50_10"].fillna(s_train["wasserstein_distance.w50_10"].mean())
    '''


    s_train["org.value"] = s_train["value"]

    '''
    s_train = mark_peaks(s_train, "value.w50.std");

    for w in [20, 50]:
        s_train["value_r2_w"+str(w)] = calculate_ratio(s_train["value"], s_train["value"].rolling(w).apply(lambda w: np.percentile(w, [50])[0], raw=False))

    s_train["value_r2_w50_wo_season"] = s_train["value"] / s_train["value"].rolling(50).apply(remove_sesonality_and_take_ratio, raw=False)
    s_train["value_r2_w50_wo_season"] = s_train["value_r2_w50_wo_season"].fillna(1)
    '''


    #s_train["value.w50.std.smooth"] = scipy.ndimage.gaussian_filter1d(s_train["value.w50.std"].values, 10)



    s_train["value.trend_removed"] = s_train["value"] - s_train["value.w50.mean"]
    seasonal_componet = s_train["value.trend_removed"].rolling(50).apply(get_seasonal_component, raw=False)
    seasonal_componet = seasonal_componet.fillna(0)
    s_train["value.season_trend_removed"] = s_train["value.trend_removed"] - seasonal_componet

    #before generating features
    s_train["value"] = s_train["value.season_trend_removed"]
    s_train = add_base_stat_features(s_train, prefix="nom.")

    #print(np.isfinite(s_train).all())
    #na_df = np.where(s_train.values >= np.finfo(np.float64).max)
    #print(np.any(na_df, axis=0))
    #na_columns = s_train.columns[na_df.any()].tolist()

    #print(na_columns)



    s_train["nom.value.w50.std.ratio"] = calculate_ratio(s_train["nom.value.w50.std"], s_train["nom.value.w50.std"].rolling(10).mean())
    s_train["nom.value.w50.kurt.ratio"] = calculate_ratio(s_train["nom.value.w50.kurt"], s_train["nom.value.w50.kurt"].rolling(10).mean())
    s_train["nom.value.w50.entropy.ratio"] = calculate_ratio(s_train["nom.value.w50.entropy"], s_train["nom.value.w50.entropy"].rolling(10).mean())

    #s_train["residual_without_seasonalitiy_via_acorr"] = s_train["value"].rolling(50).apply(residual_without_seasonalitiy_via_acorr)
    #print("residual_without_seasonalitiy_via_acorr added")

    s_train["value"] = s_train["org.value"]


    s_train = s_train.drop(["org.value"], axis=1)
    return s_train

def calculate_ratio(s1, s2):
    l = len(s1.values)
    return [s1.values[i]/s2.values[i] if s2.values[i] != 0 and not(np.isnan(s2.values[i])) else 1000 for i in range(l)]
