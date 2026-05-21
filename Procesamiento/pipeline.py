from processing import *

def emg_processing_pipeline(raw_signal, fs):
    remove_offset = offset(raw_signal)
    notch_filter = notch(remove_offset, fs)
    filtered_signal = bandpass(notch_filter, fs)
    rectified_signal = rectify(filtered_signal)
    amplitude_envelope = envelope(rectified_signal, fs)
    normalized_envelope = scale_max(amplitude_envelope)
    return {
        "raw": raw_signal,
        "filtered": filtered_signal,
        "rectified": rectified_signal,
        "envelope": amplitude_envelope,
        "normalized": normalized_envelope
    }