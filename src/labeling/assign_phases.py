from __future__ import annotations

import pandas as pd


def assign_phase_from_unity(events_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna fases a partir de los eventos del protocolo Unity."""
    out = events_df.copy()
    out["phase"] = out["event_type"].map(
        {
            "gesture_show": "gesture_prompt",
            "countdown": "countdown",
            "hold_start": "gesture",
            "rest_start": "rest",
        }
    )

    out["label"] = "Reposo"
    gesture_mask = out["phase"] == "gesture"
    out.loc[gesture_mask, "label"] = out.loc[gesture_mask, "gesture"].fillna("Gesture")

    countdown_mask = out["phase"] == "countdown"
    out.loc[countdown_mask, "label"] = "Countdown_" + out.loc[countdown_mask, "gesture"].fillna("Gesture")

    return out


def assign_intervals(events_df: pd.DataFrame) -> pd.DataFrame:
    """Genera intervalos temporales de cada fase a partir del timeline del protocolo."""
    out = assign_phase_from_unity(events_df)
    if out.empty:
        return out

    out["start_time_ns"] = out["event_time_ns"].astype("Int64")
    out["end_time_ns"] = out["event_time_ns"].shift(-1).astype("Int64")
    out["start_time_s"] = (out["start_time_ns"] / 1_000_000_000).astype(float)
    out["end_time_s"] = (out["end_time_ns"] / 1_000_000_000).astype(float)
    out["duration_s"] = out["end_time_s"] - out["start_time_s"]

    return out
