"""Filtrado de la señal EMG continua (se aplica ANTES de segmentar)."""
import numpy as np
import pandas as pd
from scipy import signal


def preprocess_emg(emg: pd.DataFrame, fs_hz: float, cfg: dict, channels: list[str]) -> pd.DataFrame:
    """Remueve DC, aplica notch y pasa-banda Butterworth (fase cero).

    Devuelve una copia de `emg` con los canales filtrados. filtfilt es no causal:
    válido para clasificación offline, no para tiempo real.
    """
    out = emg.copy()
    x = out[channels].to_numpy(dtype="float64")

    if cfg.get("remove_dc", True):
        x = x - x.mean(axis=0)

    if cfg.get("notch_hz"):
        b, a = signal.iirnotch(cfg["notch_hz"], cfg.get("notch_q", 30), fs=fs_hz)
        x = signal.sosfiltfilt(signal.tf2sos(b, a), x, axis=0)

    lo, hi = cfg["bandpass_hz"]
    hi = min(hi, 0.99 * fs_hz / 2)                       # no pasar de Nyquist
    sos = signal.butter(cfg.get("bandpass_order", 4), [lo, hi], btype="bandpass",
                        fs=fs_hz, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)

    out[channels] = x
    return out