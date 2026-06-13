from processing import *

def emg_processing_pipeline(raw_signal, fs):
    remove_offset = offset(raw_signal)
    xf_raw, mag_raw = fourier(remove_offset, fs)
    notch_filter = notch(remove_offset, fs)
    filtered_signal = bandpass(notch_filter, fs)
    xf_clean, mag_clean = fourier(filtered_signal, fs)
    rectified_signal = rectify(filtered_signal)
    amplitude_envelope = envelope(rectified_signal, fs)
    normalized_envelope = scale_max(amplitude_envelope)

    return {
        "raw": raw_signal,
        "raw xf": xf_raw,
        "raw mag": mag_raw,
        "filtered": filtered_signal,
        "clean xf": xf_clean,
        "clean mag": mag_clean,
        "rectified": rectified_signal,
        "envelope": amplitude_envelope,
        "normalized": normalized_envelope
    }