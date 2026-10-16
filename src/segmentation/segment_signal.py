"""Recorta la señal continua según los intervalos del protocolo."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Segment:
    segment_id: int
    label: str
    label_id: int
    phase: str
    trial: int
    rep: int
    t_start_ns: int
    t_end_ns: int
    data: np.ndarray                 # (n_samples, n_channels)
    valid: bool = True
    reason: str = "ok"

    @property
    def duration_s(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1e9


def segment_signal(emg: pd.DataFrame, intervals: pd.DataFrame, margins: dict,
                   channels: list[str], include_phases=("Gesto", "Reposo")) -> list[Segment]:
    """Un segmento por intervalo del protocolo (Countdown se excluye), recortando
    `onset_s` al inicio y `offset_s` al final según margins[fase]."""
    t = emg["t_ns"].to_numpy()
    x = emg[channels].to_numpy()
    segs: list[Segment] = []
    for _, r in intervals.iterrows():
        if r["phase"] not in include_phases:
            continue
        m = margins.get(r["phase"], {})
        t0 = int(r["t_start_ns"] + m.get("onset_s", 0.0) * 1e9)
        t1 = int(r["t_end_ns"] - m.get("offset_s", 0.0) * 1e9)
        if t1 <= t0:
            continue
        i0, i1 = np.searchsorted(t, [t0, t1], side="left")
        segs.append(Segment(len(segs), r["label"], int(r["label_id"]), r["phase"],
                            int(r["trial"]), int(r["rep"]), t0, t1, x[i0:i1]))
    return segs