"""Piezas compartidas por los modelos tabulares y la CNN."""
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import load_yaml

PRED_META = ["subject", "session", "segment_id", "window_idx", "rep", "trial"]


def models_root() -> Path:
    """data/analysis/models: resultados por modelo."""
    return Path(load_yaml("experiment/settings.yaml")["paths"]["analysis"]) / "models"


def undersample_indices(y, seed: int) -> np.ndarray:
    """Posiciones a conservar: cada clase se recorta a la mediana de tamaños de clase.
    Solo para ENTRENAMIENTO."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    cap = int(np.median(counts))
    rng = np.random.default_rng(seed)
    keep = []
    for c, n in zip(classes, counts):
        idx = np.flatnonzero(y == c)
        keep.append(idx if n <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(keep))


def prediction_frame(df: pd.DataFrame, test_idx, y_pred, fold_name: str) -> pd.DataFrame:
    out = df.iloc[test_idx][PRED_META].reset_index(drop=True)
    out.insert(0, "fold", fold_name)
    out["y_true"] = df["label_id"].to_numpy()[test_idx]
    out["y_pred"] = np.asarray(y_pred)
    return out