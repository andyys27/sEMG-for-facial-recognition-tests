"""
Utilidades de bajo nivel sobre la señal
Suavizado, baseline (fijo y movil), umbral doble (hysteresis) y 
metricas de forma (energia, factor de cresta)
No dependen de la logica de deteccion de eventos ni del protocolo
"""

import numpy as np
import pandas as pd


def compute_threshold(signal, time, baseline_window_sec, k):
    # Umbral fijo: mean + k*std sobre ventana de reposo inicial unica
    mask = time <= baseline_window_sec
    baseline = signal[mask]
    return float(baseline.mean() + k * baseline.std())


def smooth_signal(signal, fs, window_sec):
    # Media movil centrada
    w = max(1, int(window_sec * fs))
    return pd.Series(signal).rolling(window=w, center=True).mean().ffill().bfill().values


def rolling_baseline_stats(signal, fs, window_sec, min_frac=0.25):
    # Baseline movil robusto: mediana + MAD (Median Absolute Deviation) en ventana causal
    w = max(3, int(window_sec * fs))
    min_periods = max(3, int(w * min_frac))
    s = pd.Series(signal)

    med = s.rolling(window=w, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window=w, min_periods=min_periods).median()

    med = med.bfill().ffill().values
    mad = mad.bfill().ffill().values

    sigma = mad * 1.4826    # escala MAD equivalente a std bajo normalidad
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    return med, sigma


def hysteresis_thresholds(med, sigma, k_on, k_off_ratio):
    # Umbral de encendido (estricto) y de apagado (mas laxo)
    upper = med + k_on * sigma
    lower = med + (k_on * k_off_ratio) * sigma
    return upper, lower


def event_shape_metrics(signal_seg, sigma_seg, baseline_med_seg):

    # Metricas de forma para distinguir un gesto real de un artefacto:
    # energy_ratio. Un pico breve que apenas cruza el umbral pero decae se descarta
    # crest_factor. Un crest factor casi 1 (demasiado plano) sugiere fluctuacion de ruido
    excess = np.clip(signal_seg - baseline_med_seg, 0, None)
    energy_event = float(np.sum(excess ** 2))
    energy_noise_ref = float(np.sum(sigma_seg ** 2)) + 1e-12
    energy_ratio = energy_event / energy_noise_ref

    peak = float(signal_seg.max())
    rms = float(np.sqrt(np.mean(signal_seg ** 2))) + 1e-12
    crest_factor = peak / rms

    return energy_ratio, crest_factor