"""Valida y sincroniza EMG + eventos de Unity en una misma línea de tiempo.

Todo el proyecto usa `t_ns` (int64, ns desde epoch UTC) como reloj común:
- EMG/IMU: derivado de su columna Timestamp.
- Unity: columna t_ns (ya corregida con offset al reloj del PC).
"""
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from .load_emg import load_emg, load_imu
from .load_events import load_events


@dataclass
class Session:
    """Estructura estándar de una sesión; entrada de la capa protocol."""
    subject: str
    session: str
    emg: pd.DataFrame
    events: pd.DataFrame
    meta: dict = field(default_factory=dict)
    imu: pd.DataFrame | None = None
    report: dict = field(default_factory=dict)


def _check_clock(df: pd.DataFrame, name: str) -> dict:
    t = df["t_ns"].to_numpy()
    if len(t) < 2:
        raise ValueError(f"{name}: muy pocas muestras ({len(t)})")
    d = np.diff(t)
    if (d < 0).any():
        raise ValueError(f"{name}: timestamps no monotonos ({int((d < 0).sum())} retrocesos)")
    med = float(np.median(d[d > 0])) if (d > 0).any() else float("nan")
    return {
        "n": int(len(t)),
        "start_ns": int(t[0]),
        "end_ns": int(t[-1]),
        "duration_s": float((t[-1] - t[0]) / 1e9),
        "fs_est_hz": float(1e9 / med),
        "duplicates": int((d == 0).sum()),
        "gaps_gt_3x": int((d > 3 * med).sum()),
        "max_gap_ms": float(d.max() / 1e6),
    }


def sync_timestamps(emg: pd.DataFrame, events: pd.DataFrame, meta: dict | None = None,
                    imu: pd.DataFrame | None = None, subject: str = "", session: str = "",
                    tol_ms: float = 1.0) -> Session:
    """Valida consistencia entre fuentes y devuelve una Session con `report`.

    Errores (ValueError): relojes no monótonos o EMG y eventos sin solapamiento.
    Warnings: eventos fuera del rango del EMG, huecos en la señal y desfase entre
    `t_ns` y `Timestamp` del CSV de Unity mayor que `tol_ms`.
    """
    rep = {"emg": _check_clock(emg, "EMG")}
    if imu is not None:
        rep["imu"] = _check_clock(imu, "IMU")

    ev_t = events["t_ns"].to_numpy()
    e0, e1 = rep["emg"]["start_ns"], rep["emg"]["end_ns"]
    if ev_t.max() < e0 or ev_t.min() > e1:
        raise ValueError("EMG y eventos de Unity no se solapan: revisar reloj/zona horaria")
    outside = int(((ev_t < e0) | (ev_t > e1)).sum())
    if outside:
        warnings.warn(f"{outside} eventos de Unity caen fuera del rango del EMG")

    skew_ms = float(np.abs(events["t_ns"] - events["ts_ns"]).max() / 1e6)
    if skew_ms > tol_ms:
        warnings.warn(f"t_ns y Timestamp de Unity difieren hasta {skew_ms:.3f} ms")

    if rep["emg"]["gaps_gt_3x"]:
        warnings.warn(f"EMG con {rep['emg']['gaps_gt_3x']} huecos > 3x el periodo "
                      f"(max {rep['emg']['max_gap_ms']:.1f} ms)")

    rep["events"] = {"n": int(len(events)), "outside_emg": outside, "clock_skew_ms": skew_ms,
                     "protocol_span_s": float((ev_t.max() - ev_t.min()) / 1e9)}
    return Session(subject, session, emg, events, {} if meta is None else meta,
                   {} if imu is None else imu, rep)


def _find(dir_: Path, pattern: str) -> Path | None:
    hits = sorted(dir_.glob(pattern))
    return hits[0] if hits else None


def load_session(session_dir: str | Path, subject: str | None = None) -> Session:
    """Carga data/raw/<subject>/<session>/ y sincroniza.

    Busca: *EMG*with_timestamp.csv, Unity_events.csv, *meta.json (opcional)
    y Baiobit*.csv (opcional).
    """
    d = Path(session_dir)
    emg_p, ev_p = _find(d, "*EMG*with_timestamp.csv"), _find(d, "Unity_events.csv")
    if emg_p is None or ev_p is None:
        raise FileNotFoundError(f"{d}: se requieren *EMG*with_timestamp.csv y Unity_events.csv")
    meta_p, imu_p = _find(d, "*meta.json"), _find(d, "Baiobit*.csv")
    return sync_timestamps(
        load_emg(emg_p), load_events(ev_p),
        load_imu(imu_p) if imu_p else None,
        subject=subject or d.parent.name, session=d.name,
    )