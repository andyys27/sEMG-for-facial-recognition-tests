"""Exporta el dataset a data/processed/datasets/."""
from pathlib import Path
import numpy as np
import pandas as pd


def export_dataset(df: pd.DataFrame, raw: dict, out_dir: str | Path = "data/processed/datasets"):
    """Escribe features_dataset.csv (modelos tabulares) y raw_windows.npz (CNN)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_csv, p_raw = out / "features_dataset.csv", out / "raw_windows.npz"
    df.to_csv(p_csv, index=False)
    np.savez_compressed(p_raw, **raw)
    return p_csv, p_raw