"""
Construye un dataset de ventanas CRUDAS (tiempo x canal) a partir de
processed_labeled.csv, pensado para un CNN 1D

Por que EMG*_Filtered:
  - EMG*_Raw: Un CNN aprenderia del ruido
  - EMG*_Envelope_RMS: Es el mismo tipo de informacion que ya usan las 
    features escalares (RMS/MAV) de los modelos tabulares. El CNN no 
    aprende nada nuevo
  - EMG*_Filtered: Conserva la forma de onda y el contenido de frecuencia 
    que un CNN 1D si puede aprovechar

Se guarda en un .npz (cada ventana es una matriz 2D)
    X: (n_windows, n_samples_fixed, n_channels)
    y, groups, emotion, block, confidence, window_start_s: (n_windows,)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class RawDatasetConfig:
    signal_type: str = "Filtered"   # "Filtered" | "Raw" | "Envelope_RMS" | "Normalized_Z"
    window_sec: float = 0.25
    overlap: float = 0.5
    exclude_countdown: bool = True
    include_reposo: bool = True
    n_samples_fixed: int = 64       # puntos por ventana tras remuestrear
    min_window_samples: int = 10


def _resample_window(window_arr, n_samples_fixed):
    win_samples = window_arr.shape[0]
    if win_samples == n_samples_fixed:
        return window_arr
    x_old = np.linspace(0, 1, win_samples)
    x_new = np.linspace(0, 1, n_samples_fixed)
    out = np.zeros((n_samples_fixed, window_arr.shape[1]))
    for c in range(window_arr.shape[1]):
        out[:, c] = np.interp(x_new, x_old, window_arr[:, c])
    return out


def _windows_for_segment(seg_df, cfg, raw_cols, fs, subject_id, label, block, emotion, confidence):
    win_samples = max(1, int(cfg.window_sec * fs))
    step_samples = max(1, int(win_samples * (1 - cfg.overlap)))

    if len(seg_df) < max(win_samples, cfg.min_window_samples):
        return []

    raw_values = seg_df[raw_cols].values
    time_values = seg_df["Time_Seconds"].values

    rows = []
    start = 0
    while start + win_samples <= len(seg_df):
        window_arr = raw_values[start:start + win_samples, :]
        window_resampled = _resample_window(window_arr, cfg.n_samples_fixed)
        rows.append({
            "X": window_resampled,
            "Subject": subject_id,
            "Label": label,
            "Emotion": emotion,
            "Block": block,
            "Confidence": confidence,
            "Window_Start_s": float(time_values[start]),
        })
        start += step_samples

    return rows


def build_raw_windows_from_labeled_csv(csv_path, subject_id, channel_order, cfg: RawDatasetConfig = None):
    # channel_order: lista de nombres BASE de canal en orden canonico entre sujetos
    
    if cfg is None:
        cfg = RawDatasetConfig()

    raw_cols = [f"{ch}_{cfg.signal_type}" for ch in channel_order]

    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: columnas no encontradas {missing}. "
                          f"Columnas disponibles: {list(df.columns)}")

    time = df["Time_Seconds"].values
    fs = 1.0 / np.mean(np.diff(time))

    if not cfg.include_reposo:
        df = df[df["Phase"] != "Reposo"].reset_index(drop=True)
    if cfg.exclude_countdown:
        df = df[df["Phase"] != "Countdown"].reset_index(drop=True)

    if df.empty:
        return []

    df = df.reset_index(drop=True)
    same_as_prev = (df["Label"] == df["Label"].shift(1)) & (df["Block"] == df["Block"].shift(1))
    segment_id = (~same_as_prev).cumsum()

    all_rows = []
    for _, seg_df in df.groupby(segment_id):
        label = seg_df["Label"].iloc[0]
        block = int(seg_df["Block"].iloc[0])
        emotion = seg_df["Emotion"].iloc[0]
        confidence = seg_df["Confidence"].iloc[0] if "Confidence" in seg_df.columns else "Alta"
        rows = _windows_for_segment(seg_df, cfg, raw_cols, fs, subject_id, label, block, emotion, confidence)
        all_rows.extend(rows)

    return all_rows


def build_raw_dataset(subject_specs, cfg: RawDatasetConfig = None, output_path=None):
    # subject_specs: dict {subject_id: (csv_path, channel_order)} donde channel_order
    # es la lista de canales BASE en orden canonico para ese sujeto 
    # Tambien acepta (csv_path, channel_order, cfg_especifico) si algun sujeto 
    # necesita una config distinta 

    if cfg is None:
        cfg = RawDatasetConfig()

    all_rows = []
    for subject_id, spec in subject_specs.items():
        if len(spec) == 3:
            csv_path, channel_order, subj_cfg = spec
        else:
            csv_path, channel_order = spec
            subj_cfg = cfg

        print(f"  Procesando {subject_id}: {csv_path}")
        print(f"    channel_order: {channel_order}  (signal_type={subj_cfg.signal_type})")
        rows = build_raw_windows_from_labeled_csv(csv_path, subject_id, channel_order, subj_cfg)
        print(f"    -> {len(rows)} ventanas crudas generadas")
        all_rows.extend(rows)

    if not all_rows:
        print("  ⚠ No se genero ninguna ventana.")
        return None

    X = np.stack([r["X"] for r in all_rows])             # (n, n_samples_fixed, n_channels)
    y = np.array([r["Label"] for r in all_rows])
    groups = np.array([r["Subject"] for r in all_rows])
    emotion = np.array([r["Emotion"] for r in all_rows])
    block = np.array([r["Block"] for r in all_rows])
    confidence = np.array([r["Confidence"] for r in all_rows])
    window_start_s = np.array([r["Window_Start_s"] for r in all_rows])

    print(f"\n  Shape final: X={X.shape}, y={y.shape}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=X, y=y, groups=groups, emotion=emotion, block=block, confidence=confidence, window_start_s=window_start_s)
        print(f"  -> Dataset crudo guardado: {out}")

    return {"X": X, "y": y, "groups": groups, "emotion": emotion, "block": block,
             "confidence": confidence, "window_start_s": window_start_s}