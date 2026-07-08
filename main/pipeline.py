from processing import *

def emg_processing_pipeline(raw_signal, fs):
    # Remover desfase de corriente continua
    clean_offset = remove_offset(raw_signal)
    xf_raw, mag_raw = fourier(clean_offset, fs)

    # Filtrado Notch y Bandpass
    notched = notch(clean_offset, fs)
    filtered_signal = bandpass(notched, fs)
    xf_clean, mag_clean = fourier(filtered_signal, fs)

    # Envolvente de activacion
    amplitude_envelope = envelope_rms(filtered_signal, fs)

    # Normalizacion por Z Score
    normalized_envelope = scale_zscore(amplitude_envelope)

    return {
        "raw": raw_signal,
        "raw xf": xf_raw,
        "raw mag": mag_raw,
        "filtered": filtered_signal,
        "clean xf": xf_clean,
        "clean mag": mag_clean,
        "envelope": amplitude_envelope,
        "normalized": normalized_envelope
    }