"""
Construye el dataset de features listo para entrenar, a partir de processed_labeled.csv

Ventana deslizante DENTRO de cada tramo contiguo de la misma etiqueta (Label) y del mismo 
evento (Block+Emotion)

Enfoque estandar en clasificacion de gestos EMG (Hudgins et al. 1993 usa 256ms/~doble de 
overlap)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path

from .features import compute_window_features


@dataclass
class DatasetConfig:
    channel_groups: dict = field(default_factory=lambda: {
        "Grupo_A": ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS"],
        "Grupo_B": ["EMG3_Envelope_RMS", "EMG4_Envelope_RMS"],
    })
    window_sec: float = 0.25        # 250 ms, estandar en literatura EMG
    overlap: float = 0.5            # 50% de traslape entre ventanas
    exclude_countdown: bool = True  # El countdown es transicion, se descarta
    include_reposo: bool = True     # Incluir la clase "Reposo" en el dataset
    min_window_samples: int = 10    # Descarta tramos demasiado cortos para 1 ventana


def _windows_for_segment(seg_df, cfg: DatasetConfig, fs, subject_id, label, block, emotion, confidence):
    # Genera las ventanas deslizantes de UN tramo contiguo y calcula sus features
    win_samples = max(1, int(cfg.window_sec * fs))
    step_samples = max(1, int(win_samples * (1 - cfg.overlap)))

    if len(seg_df) < max(win_samples, cfg.min_window_samples):
        return []

    rows = []
    start = 0
    while start + win_samples <= len(seg_df):
        window = seg_df.iloc[start:start + win_samples]
        feats = compute_window_features(window, cfg.channel_groups)
        feats["Subject"] = subject_id
        feats["Label"] = label
        feats["Emotion"] = emotion
        feats["Block"] = block
        feats["Confidence"] = confidence
        feats["Window_Start_s"] = float(window["Time_Seconds"].iloc[0])
        rows.append(feats)
        start += step_samples

    return rows


def build_windows_from_labeled_csv(csv_path, subject_id, cfg: DatasetConfig = None):
    # Lee un processed_labeled.csv y devuelve un DataFrame de features, una fila por ventana
    if cfg is None:
        cfg = DatasetConfig()

    df = pd.read_csv(csv_path, low_memory=False)
    time = df["Time_Seconds"].values
    fs = 1.0 / np.mean(np.diff(time))

    if not cfg.include_reposo:
        df = df[df["Phase"] != "Reposo"].reset_index(drop=True)
    if cfg.exclude_countdown:
        df = df[df["Phase"] != "Countdown"].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # Identificar tramos contiguos: mismo Label + mismo Block, sin huecos de indice
    df = df.reset_index(drop=True)
    same_as_prev = (df["Label"] == df["Label"].shift(1)) & (df["Block"] == df["Block"].shift(1))
    segment_id = (~same_as_prev).cumsum()

    all_rows = []
    for _, seg_df in df.groupby(segment_id):
        label = seg_df["Label"].iloc[0]
        block = int(seg_df["Block"].iloc[0])
        emotion = seg_df["Emotion"].iloc[0]
        confidence = seg_df["Confidence"].iloc[0] if "Confidence" in seg_df.columns else "Alta"
        rows = _windows_for_segment(seg_df, cfg, fs, subject_id, label, block, emotion, confidence)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def build_dataset(subject_csv_map, cfg: DatasetConfig = None, output_path=None):
    # Devuelve el DataFrame combinado de todos los sujetos y lo guarda en output_path.
    if cfg is None:
        cfg = DatasetConfig()

    frames = []
    for subject_id, csv_path in subject_csv_map.items():
        print(f"  Procesando {subject_id}: {csv_path}")
        wdf = build_windows_from_labeled_csv(csv_path, subject_id, cfg)
        print(f"    -> {len(wdf)} ventanas generadas")
        frames.append(wdf)

    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if output_path and not dataset.empty:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(out, index=False)
        print(f"\n  -> Dataset guardado: {out} ({len(dataset)} filas, {dataset.shape[1]} columnas)")

    if not dataset.empty:
        print("\n  Distribucion de clases (Label):")
        print(dataset["Label"].value_counts().to_string())
        print("\n  Distribucion de confianza:")
        print(dataset["Confidence"].value_counts().to_string())

    return dataset