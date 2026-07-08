import numpy as np
from scipy.signal import butter, iirnotch, filtfilt
from scipy.fft import fft, fftfreq

# Remover offset
def remove_offset(signal):
    return signal - np.mean(signal)

# Butterworth pasa-banda
def bandpass(signal, fs, low = 20, high = 450, order = 4):
    nyq = fs / 2
    lowcut = low / nyq
    highcut = high / nyq
    b_notch, a_notch = butter(order, [lowcut, highcut], btype="band")
    return filtfilt(b_notch, a_notch, signal)

# IIR notch para ruido de linea
def notch(signal, fs, freq = 60, Q = 30):    
    w0 = freq / (fs / 2)
    b_notch, a_notch = iirnotch(w0, Q)
    return filtfilt(b_notch, a_notch, signal)

# Normalizacion Z Score
def scale_zscore(signal):
    std = np.std(signal)
    return (signal - np.mean(signal)) / std

# Envolvente RMS
def envelope_rms(signal, fs, window_ms=50):
    window_samples = int((window_ms / 1000) * fs)
    env = np.sqrt(np.convolve(signal**2, np.ones(window_samples)/window_samples, mode='same'))
    return env

def fourier(signal, fs):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    mag = 2.0 / n * np.abs(yf[0:n//2])
    return xf, mag