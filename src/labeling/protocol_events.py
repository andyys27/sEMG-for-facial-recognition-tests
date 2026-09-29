from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_event_types(events_df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los eventos del protocolo experimental a una sola nomenclatura."""
    out = events_df.copy()
    out["event_type"] = out["event_type"].astype(str)

    mapping = {
        "gesture_show": "gesture_show",
        "countdown_3": "countdown",
        "countdown_2": "countdown",
        "countdown_1": "countdown",
        "hold_start": "hold_start",
        "rest_start": "rest_start",
    }
    out["event_type"] = out["event_type"].map(mapping).fillna(out["event_type"])
    return out


def build_event_timeline(unity_events: pd.DataFrame) -> pd.DataFrame:
    """Genera una línea de tiempo de protocolo a partir de Unity events."""
    events = normalize_event_types(unity_events)
    events = events.sort_values(["event_time_ns", "event_type"]).reset_index(drop=True)
    return events


def iter_trials(events_df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """Yield por trial para facilitar validación o diagnóstico."""
    for _, group in events_df.groupby(["trial"], dropna=False):
        yield group.sort_values("event_time_ns").reset_index(drop=True)
