"""Junta ventanas de todas las sesiones en un dataset de features + ventanas crudas."""
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import load_yaml
from ..features import extract_features

META_COLS = ["subject", "session", "segment_id", "window_idx", "trial", "rep", "label", "label_id"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def build_dataset(segments_dir: str | Path = "data/processed/segments",
                  features_cfg: dict | None = None, classes_cfg: dict | None = None):
    """Lee los *_windows.npz de `segments_dir` (uno por sesión).

    Devuelve (df, raw): `df` con META_COLS + features (una fila por ventana) y `raw`
    con las ventanas crudas alineadas fila a fila (para la CNN).
    """
    fcfg = features_cfg or load_yaml("dataset/features.yaml")
    ccfg = classes_cfg or load_yaml("dataset/classes.yaml")
    files = sorted(Path(segments_dir).glob("*_windows.npz"))
    if not files:
        raise FileNotFoundError(f"No hay *_windows.npz en {segments_dir}")

    frames, raws, channels_ref = [], [], None
    for f in files:
        z = np.load(f, allow_pickle=False)
        channels, fs = [str(c) for c in z["channels"]], float(z["fs_hz"])
        if channels_ref is None:
            channels_ref = channels
        elif channels != channels_ref:
            raise ValueError(f"{f.name}: canales {channels} != {channels_ref}")

        feats = extract_features(z["windows"], fs, channels, fcfg)
        meta = pd.DataFrame({
            "subject": str(z["subject"]), "session": str(z["session"]),
            "segment_id": z["segment_id"], "trial": z["trial"], "rep": z["rep"],
            "label": z["label"], "label_id": z["label_id"]})
        meta["window_idx"] = meta.groupby("segment_id").cumcount()
        frames.append(pd.concat([meta[META_COLS[:3] + ["window_idx", "trial", "rep", "label", "label_id"]],
                                 feats], axis=1))
        raws.append(z["windows"])

    df = pd.concat(frames, ignore_index=True)
    raw_w = np.concatenate(raws)
    keep = df["label"].isin(ccfg["use_classes"]).to_numpy()
    df, raw_w = df[keep].reset_index(drop=True), raw_w[keep]
    meta = {c: df[c].to_numpy() for c in META_COLS}
    meta = {c: (v.astype(str) if v.dtype == object else v) for c, v in meta.items()}   # texto -> unicode (sin pickle)
    raw = {"windows": raw_w, "channels": np.array(channels_ref), **meta}
    return df, raw