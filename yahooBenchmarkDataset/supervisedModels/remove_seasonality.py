import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

def create_sin_wave(frequency, num_samples, sampling_rate=10, magnitude=1):
    sine_wave = [magnitude * np.sin(2 * np.pi * frequency * x1/ sampling_rate) for x1 in range(num_samples)]
    return sine_wave

sine_wave1 = np.array(create_sin_wave(frequency=1, num_samples=1000, sampling_rate=500, magnitude = 5))
sine_wave2 = np.array(create_sin_wave(frequency=20, num_samples=1000, sampling_rate=500, magnitude = 1))
sine_wave3 = np.array(create_sin_wave(frequency=50, num_samples=1000, sampling_rate=500, magnitude = 1))
sine_wave4 = np.array(create_sin_wave(frequency=100, num_samples=1000, sampling_rate=500, magnitude = 1))
sine_wave5 = np.array(create_sin_wave(frequency=200, num_samples=1000, sampling_rate=500, magnitude = 1))

sine_wave = sine_wave1 + sine_wave3 #+ sine_wave3 #+ sine_wave4 + sine_wave5
'''Plot: Signal with seasonality'''
ax = plt.subplot(6, 1, 1)
plt.plot(sine_wave)

# get top "no_of_seasons" seasons
no_of_seasons=1   # Here we are assuming there is only one seasonality
series = sine_wave

# Compute FFT
series_fft = fft(series)

# Compute the power
power = np.abs(series_fft)
'''Plot: Power at each frequency'''
ax = plt.subplot(6, 1, 2)
plt.plot(power)

# Get the corresponding frequencies
sample_periods = fftfreq(series_fft.size)

# Find the peak frequency: we only need the positive frequencies
pos_mask = np.where(sample_periods > 0)
#print(pos_mask)
periods = sample_periods[pos_mask]
powers = power[pos_mask]
'''Plot: Considering only positive frequencies'''
ax = plt.subplot(6, 1, 3)
plt.plot(powers)

# find top frequencies and corresponding time periods for seasonal pattern
top_powers = np.argpartition(powers, -no_of_seasons)[-no_of_seasons:]

time_periods_from_fft = 1 / periods[top_powers]
time_periods = time_periods_from_fft.astype(int)
frequencies = ((series_fft.size)/time_periods).astype(int)
print(frequencies)

# Daily and Weekly seasons expected values from DF
freq_expected = [0]

# One of the seasonality returned from FFT should be within range of Expected time period
for frequency in freq_expected:
    nearest_frequency = frequencies.flat[np.abs(frequencies - frequency).argmin()]
    print(nearest_frequency)
    '''Plot: FFT before removing seasonality'''
    ax = plt.subplot(6, 1, 4)
    plt.plot(series_fft)
    series_fft[int(series_fft.size - nearest_frequency)] = 0
    series_fft[int(nearest_frequency)] = 0
    '''Plot: FFT after removing seasonality'''
    ax = plt.subplot(6, 1, 5)
    plt.plot(series_fft)

recovered = np.fft.ifft(series_fft)
'''Plot: Seasonality removed signal'''
ax = plt.subplot(6, 1, 6)
plt.plot(recovered)
plt.show()
