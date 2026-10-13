"""Etiqueta cada muestra EMG a partir de los intervalos del protocolo."""
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import load_yaml
from .parse_events import parse_events, protocol_params
from .assign_phases import assign_phases
from .map_emotions import map_emotions


def build_labels(t_ns, intervals: pd.DataFrame, outside_label: str | None = None,
                 outside_id: int | None = None) -> pd.DataFrame:
    """Una fila por muestra: t_ns, label, label_id, phase, trial, rep, interval_idx.
    Intervalos semiabiertos [t_start, t_end). Fuera del protocolo -> 'Fuera_protocolo'."""
    outside_label = outside_label or load_yaml("protocol/phases.yaml")["outside_label"]
    if outside_id is None:
        outside_id = load_yaml("experiment/labels.yaml")["special"][outside_label]
    t = np.asarray(t_ns, dtype="int64")
    iv = intervals.sort_values("t_start_ns").reset_index(drop=True)
    starts, ends = iv["t_start_ns"].to_numpy(), iv["t_end_ns"].to_numpy()

    idx = np.searchsorted(starts, t, side="right") - 1
    idx_c = np.clip(idx, 0, None)
    inside = (idx >= 0) & (t < ends[idx_c])

    def pick(col, fill):
        return np.where(inside, iv[col].to_numpy()[idx_c], fill)

    return pd.DataFrame({
        "t_ns": t,
        "label": pick("label", outside_label),
        "label_id": pick("label_id", outside_id).astype("int64"),
        "phase": pick("phase", outside_label),
        "trial": pick("trial", -1).astype("int64"),
        "rep": pick("rep", -1).astype("int64"),
        "interval_idx": np.where(inside, idx, -1),
    })


def label_session(session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session -> (intervals, labels por muestra EMG)."""
    marks = parse_events(session.events)
    params = protocol_params(session.events)
    end_ns = int(session.emg["t_ns"].iloc[-1])
    intervals = map_emotions(assign_phases(marks, params, end_ns=end_ns))
    return intervals, build_labels(session.emg["t_ns"], intervals)


def export_labels(session, intervals: pd.DataFrame, labels: pd.DataFrame,
                  out_dir: str | Path = "data/processed/labels") -> tuple[Path, Path]:
    """Escribe <subject>_<session>_intervals.csv y _labels.csv (unible por t_ns)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{session.subject}_{session.session}"
    p_iv, p_lb = out / f"{stem}_intervals.csv", out / f"{stem}_labels.csv"
    intervals.to_csv(p_iv, index=False)
    labels.to_csv(p_lb, index=False)
    return p_iv, p_lb