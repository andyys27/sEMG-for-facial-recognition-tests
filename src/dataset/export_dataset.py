"""Exporta y vuelve a cargar el dataset en data/processed/datasets/."""
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


def load_dataset(dir_: str | Path = "data/processed/datasets", with_raw: bool = True):
    """Carga (df, raw). `raw` es None si with_raw=False. Verifica que las ventanas crudas
    estén alineadas fila a fila con la tabla de features."""
    d = Path(dir_)
    df = pd.read_csv(d / "features_dataset.csv")
    if not with_raw:
        return df, None
    raw = dict(np.load(d / "raw_windows.npz", allow_pickle=False))
    if not (np.array_equal(raw["segment_id"], df["segment_id"].to_numpy())
            and np.array_equal(raw["window_idx"], df["window_idx"].to_numpy())):
        raise ValueError("raw_windows.npz y features_dataset.csv no están alineados; reconstruir con prepare_data")
    return df, raw