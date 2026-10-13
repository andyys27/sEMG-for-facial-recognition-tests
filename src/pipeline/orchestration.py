"""Encadena las etapas del flujo. No contiene lógica técnica propia."""
from pathlib import Path
import pandas as pd

from ..config import load_yaml
from ..io import load_session
from ..protocol import label_session
from ..segmentation import (preprocess_emg, segment_signal, validate_segments,
                            windows_from_segments, save_windows)

STAGES = ["load", "sync", "protocol", "segmentation", "features", "dataset", "train", "evaluate"]


def process_session(session_dir: str | Path, out_dir: str | Path = "data/processed/segments") -> dict:
    """Sesión cruda -> ventanas etiquetadas (.npz) + tabla de segmentos (.csv)."""
    s = load_session(session_dir)
    intervals, _ = label_session(s)

    cfg = load_yaml("experiment/settings.yaml")
    wcfg = load_yaml("dataset/windows.yaml")
    channels, fs = cfg["channel_names"], s.report["emg"]["fs_est_hz"]

    filtered = preprocess_emg(s.emg, fs, cfg["preprocessing"], channels)
    segs = segment_signal(filtered, intervals, cfg["margins"], channels)
    segs = validate_segments(segs, fs, wcfg["min_segment_ms"] / 1000)
    data = windows_from_segments(segs, fs, wcfg["window_ms"], wcfg["overlap"])

    stem = f"{s.subject}_{s.session}"
    n_win = pd.Series(data["segment_id"]).value_counts()
    table = pd.DataFrame([{
        "segment_id": g.segment_id, "label": g.label, "phase": g.phase, "trial": g.trial,
        "rep": g.rep, "t_start_ns": g.t_start_ns, "t_end_ns": g.t_end_ns,
        "n_samples": len(g.data), "valid": g.valid, "reason": g.reason,
        "n_windows": int(n_win.get(g.segment_id, 0))} for g in segs])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{stem}_segments.csv", index=False)
    save_windows(out / f"{stem}_windows.npz", data, channels=channels, fs_hz=fs,
                 subject=s.subject, session=s.session)
    return {"stem": stem, "fs": fs, "segments": table, "windows": data, "dir": out}