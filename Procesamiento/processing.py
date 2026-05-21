import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

# Remover offset
def offset(signal):
    return signal - np.mean(signal)

# Butterworth pasa-banda
def bandpass(signal, fs):
    low = 20 
    high = 450
    order = 4
    nyq = fs / 2
    lowcut = low / nyq
    highcut = high / nyq
    b_notch, a_notch = butter(order, [lowcut, highcut], btype="band")
    return filtfilt(b_notch, a_notch, signal)

# IIR notch para ruido de linea
def notch(signal, fs):
    freq = 60.0
    Q = 30.0
    w0 = freq / (fs / 2)
    b_notch, a_notch = iirnotch(w0, Q)
    return filtfilt(b_notch, a_notch, signal)

# Rectificar senal
def rectify(signal):
    return np.abs(signal)

# Normalizacion max
def scale_max(signal):
    mvc = signal / np.max(np.abs(signal))
    return mvc

# Envolvente RMS
def envelope(signal, fs):
    window_ms = 50
    window_samples = int((window_ms / 1000) * fs)
    env = np.sqrt(np.convolve(signal**2, np.ones(window_samples)/window_samples, mode='same'))
    return env