from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TimeSyncResult:
    emg_df: pd.DataFrame
    unity_df: pd.DataFrame
    aligned_df: pd.DataFrame
    offset_ns: int | None = None


def align_emg_and_unity(emg_df: pd.DataFrame, unity_df: pd.DataFrame, time_col: str = "time_s") -> TimeSyncResult:
    """Sincroniza la señal EMG con la línea de tiempo de Unity.

    Este es un esqueleto de trabajo: el objetivo es dejar la base para
    convertir ambos datos a un mismo marco temporal antes del etiquetado.
    """
    emg = emg_df.copy()
    unity = unity_df.copy()

    if "time_s" not in emg.columns:
        emg["time_s"] = emg.index.to_numpy(dtype=float) / max((1 if len(emg) == 0 else 1), 1)

    unity = unity.sort_values("event_time_ns").reset_index(drop=True)
    if len(unity) == 0:
        raise ValueError("Unity events no contiene filas válidas para sincronizar.")

    offset_ns = int(unity["event_time_ns"].iloc[0])
    unity["time_s_from_unity"] = (unity["event_time_ns"] - offset_ns) / 1_000_000_000.0

    aligned = emg.copy()
    aligned["time_s_unity_reference"] = aligned[time_col].astype(float)

    return TimeSyncResult(
        emg_df=emg,
        unity_df=unity,
        aligned_df=aligned,
        offset_ns=offset_ns,
    )


def load_and_sync(unity_csv_path: str | Path, emg_csv_path: str | Path) -> TimeSyncResult:
    from src.io.load_unity_events import load_unity_events

    unity_df = load_unity_events(unity_csv_path)
    emg_df = pd.read_csv(emg_csv_path)
    return align_emg_and_unity(emg_df, unity_df)
