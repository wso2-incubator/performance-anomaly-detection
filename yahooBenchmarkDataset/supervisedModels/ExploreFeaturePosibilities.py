
from wso2.time_series import *
from wso2tools.plot_templates import *
from statsmodels.tsa.seasonal import seasonal_decompose


def copy_org(df):
    df["value.org"] = df["value"]
    return df

def load_yahoo_dataset2(feature_fn=create_timeseries_features, file_list=None):
    dir = "/Users/srinath/playground/Datasets/SystemsData/YahooAnomalyDataset/A1Benchmark/"
    if file_list is None:
        file_list = [ 11, 19, 15, 9, 45, 56, 59, 64, 41, 16, 54, 39, 2, 44, 23, 18, 47, 31, 24, 49, 26, 3, 0, 60, 52, 40, 43, 61, 12, 17, 33, 20, 1, 38, 7, 50, 30, 32, 21, 48, 5, 35, 4, 14, 29, 22, 55, 8, 62, 6, 58, 36, 53, 34, 10, 63, 28, 51, 25, 42, 57, 46, 13, 37, 27]

    data_sets = [pd.read_csv(dir+"real_" +str(i+1)+".csv") for i in file_list]
    normalized_datasets = [normalize_value(copy_org(df)) for df in data_sets]
    transformed_datasets = [feature_fn(df) for df in normalized_datasets]
    data_set = pd.concat(transformed_datasets)

    data_set = data_set.fillna(0)
    data_set= data_set.drop(["timestamp"], axis=1)

    X = data_set
    y = data_set.pop("is_anomaly")
    return X, y



'''

Sesonality options 
    https://machinelearningmastery.com/time-series-seasonality-with-python/#:~:text=A%20simple%20way%20to%20correct,the%20value%20from%20last%20week.
    #https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy

'''

def remove_trend_sesonality(s_train):
    '''
    :param s_train:
    :return:
    '''
    window_list = [5, 25, 50]

    for i, w in enumerate(window_list):
        s_train["value.w" + str(w) + ".mean"] = s_train["value"].rolling(w).mean()\
            .fillna(s_train["value"] if i ==0 else s_train["value.w" + str(window_list[i-1]) + ".mean"])

    s_train["value.trend_removed"] = s_train["value"] - s_train["value.w50.mean"]

    seasonal_componet = s_train["value.trend_removed"].rolling(50).apply(get_seasonal_component, raw=False)
    seasonal_componet = seasonal_componet.fillna(0)
    s_train["seasonal_componet"] = seasonal_componet
    s_train["value.season_trend_removed"] = s_train["value.trend_removed"] - seasonal_componet

    return s_train

def explore_features_in_ts():
    #[59, 65, 58, 62]
    data_set = [
        [[41], "trend_change"],
        [[55], "spikes"],
        [[50, 65], "season"],
        [[10], "level_shift"]

     ]

    for [d, name] in data_set:
        X, y =  load_yahoo_dataset2(file_list=d, feature_fn=remove_trend_sesonality)
        #features = ['value', 'value.w50.std', 'value.w50.kurt', "value.w50.std.ratio", "value.w50.std.smooth", "value.w50.kurt.ratio", "value.w50.std.peaks"
        #            , "value_r2_w50_wo_season", "value.trend_removed", "value.season_trend_removed"]
        #features = ['value', 'nom.value.w50.std', 'nom.value.w50.kurt', "nom.value.w50.std.ratio", "nom.value.w50.kurt.ratio",
        #            "value.season_trend_removed"]
        features = ['value.org', 'value', 'value.season_trend_removed', "seasonal_componet"]
        plot_waves([X[f].values for f in features], features, "../output/"+name+".png", labels=y)

def create_sin_wave(frequency, num_samples, sampling_rate=10):
    sine_wave = [np.sin(2 * np.pi * x1/ frequency) for x1 in range(num_samples)]
    return sine_wave


def test_sesonality():
    sine_wave1 = np.array(create_sin_wave(frequency=10, num_samples=1000, sampling_rate=50))
    sine_wave2 = np.array(create_sin_wave(frequency=500, num_samples=1000, sampling_rate=50))
    sine_wave = sine_wave1 + sine_wave2

    df = pd.DataFrame()
    df["value"] = sine_wave
    df = remove_trend_sesonality(df)

    features = ['value', 'value.season_trend_removed', "seasonal_componet"]
    plot_waves([df[f].values for f in features], features, "../output/sine.png")

def seasonal_decompose():
    sine_wave1 = np.array(create_sin_wave(frequency=10, num_samples=1000, sampling_rate=50))
    sine_wave2 = np.array(create_sin_wave(frequency=500, num_samples=1000, sampling_rate=50))
    w = sine_wave1 + sine_wave2

    value = pd.Series(w)
    value.index = pd.DatetimeIndex(range(len(w)))

    result = seasonal_decompose(value, model='additive')
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    values = [result.observed, result.trend, result.seasonal, result.resid]
    names = ["value", "result.trend", "result.seasonal", "result.resid"]

    plot_waves(values, names, "../output/sine2.png")


def test_get_seasonality_with_fft():
    sine_wave1 = np.array(create_sin_wave(frequency=10, num_samples=1000, sampling_rate=50))
    sine_wave2 = np.array(create_sin_wave(frequency=500, num_samples=1000, sampling_rate=50))
    w = sine_wave1 + sine_wave2 #+ np.random.normal(size=1000)/10


    frequencies = np.fft.fft(w)

    #TODO I am taking largest real value, is that right
    freq_size = np.sqrt(frequencies.real*frequencies.real + frequencies.imag * frequencies.imag)

    sortedIndexs = np.argsort(freq_size) #returns values in assending order

    index2keep = -1
    n = len(sortedIndexs)
    for i in range(n):
        if sortedIndexs[n -1 -i] >= n/2:
            index2keep = sortedIndexs[n -1 -i]
            break

    period_freq = np.zeros(n, dtype=complex)
    period_freq[index2keep] = frequencies[index2keep]
    period_freq_size = np.sqrt(period_freq.real*period_freq.real + period_freq.imag*period_freq.imag)
    recovered_signal = np.fft.ifft(period_freq)
    x2 = np.concatenate([recovered_signal, recovered_signal])
    values = [w, freq_size, period_freq_size, recovered_signal, x2]
    names =["value", "freqs", "period_freq_size", "recovered_signal", "2x"]

    plot_waves(values, names, "../output/sine2.png")

def test_sesonality_with_autocorr():
    dir = "/Users/srinath/playground/Datasets/SystemsData/YahooAnomalyDataset/A1Benchmark/"
    file_list = list(range(1, 67))

    data_sets = [pd.read_csv(dir + "real_" + str(i) + ".csv") for i in file_list]

    plot_data = []

    for i, df in enumerate(data_sets):

        x = df["value"]
        x = x/ np.percentile(x, [90])[0]
        value = x
        mean_w50 = x.rolling(50).mean().fillna(x.mean())
        n = x.size
        x = x/ mean_w50
        norm = (x - np.mean(x))
        result = np.correlate(norm, norm, mode='same')
        acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
        lag = np.abs(acorr).argmax() + 1
        r = acorr[lag - 1]
        if np.abs(r) > 0.5:
            print("dataset_" + str(i+1) + 'Appears to be autocorrelated with r = {}, lag = {}'.format(r, lag))
            #plot_data.append((value, acorr, "dataset_" + str(i+1)))
            plot_data.append((value, x, "dataset_" + str(i + 1)))
        else:
            print("dataset_" + str(i+1) + ' Appears to be not autocorrelated acorr=', r)

    plot_insame(plot_data, "autocorr.png")



#explore_features_in_ts()
test_get_seasonality_with_fft()

test_sesonality_with_autocorr()
