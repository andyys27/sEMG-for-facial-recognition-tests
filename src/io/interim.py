"""Datos intermedios (data/interim): señal filtrada y eventos parseados por sesión."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import load_yaml


def interim_dir(subject: str, session: str, root: str | Path | None = None) -> Path:
    root = root or load_yaml("experiment/settings.yaml")["paths"]["interim"]
    return Path(root) / subject / session


def cache_key(preprocessing_cfg: dict, fs_hz: float, n_samples: int, t0_ns: int) -> str:
    """Huella de lo que determina la señal filtrada: config de filtros + datos de origen.
    Si cambia cualquiera, el archivo intermedio se considera obsoleto y se recalcula."""
    payload = json.dumps({"p": preprocessing_cfg, "fs": round(fs_hz, 1), "n": n_samples, "t0": t0_ns},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def save_filtered(directory: str | Path, filtered: pd.DataFrame, channels: list[str], key: str) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "emg_filtered.npz"
    np.savez_compressed(p, t_ns=filtered["t_ns"].to_numpy(), data=filtered[channels].to_numpy(),
                        channels=np.array(channels), key=key)
    return p


def load_filtered(directory: str | Path, channels: list[str], key: str) -> pd.DataFrame | None:
    """Devuelve la señal filtrada guardada, o None si no existe o está obsoleta."""
    p = Path(directory) / "emg_filtered.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if str(z["key"]) != key or [str(c) for c in z["channels"]] != list(channels):
        return None
    df = pd.DataFrame(z["data"], columns=list(channels))
    df.insert(0, "t_ns", z["t_ns"])
    return df


def save_session_artifacts(directory: str | Path, marks: pd.DataFrame, sync_report: dict) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    marks.to_csv(d / "events_parsed.csv", index=False)
    (d / "sync_report.json").write_text(json.dumps(sync_report, indent=2), encoding="utf-8")