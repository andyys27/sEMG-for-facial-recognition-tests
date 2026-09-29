from __future__ import annotations

import pandas as pd
from pathlib import Path


def _parse_detail(detail: str) -> dict:
    if pd.isna(detail) or detail == "":
        return {}

    parsed: dict[str, str] = {}
    for item in str(detail).split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def load_unity_events(csv_path: str | Path) -> pd.DataFrame:
    """Carga el CSV de Unity y normaliza los eventos del protocolo."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["event_time_ns"] = df["t_unity_ns"].astype("Int64")
    df["event_type"] = df["label"].astype(str)
    df["gesture"] = df["detail"].map(lambda x: _parse_detail(x).get("gesture", ""))
    df["gesture_idx"] = df["detail"].map(lambda x: _parse_detail(x).get("gesture_idx", "")).replace({"": None})
    df["trial"] = df["detail"].map(lambda x: _parse_detail(x).get("trial", "")).replace({"": None})
    df["rep"] = df["detail"].map(lambda x: _parse_detail(x).get("rep", "")).replace({"": None})

    # Convierte columnas numéricas a enteros cuando sea posible
    for col in ["trial", "rep", "gesture_idx"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["phase"] = df["event_type"].map(
        {
            "gesture_show": "gesture_prompt",
            "countdown_3": "countdown",
            "countdown_2": "countdown",
            "countdown_1": "countdown",
            "hold_start": "gesture",
            "rest_start": "rest",
        }
    )

    return df.sort_values("event_time_ns").reset_index(drop=True)


def extract_trial_sequence(events_df: pd.DataFrame) -> pd.DataFrame:
    """Expone la secuencia de trial/gesture para análisis de protocolo."""
    out = events_df[["event_time_ns", "event_type", "gesture", "trial", "rep", "phase"]].copy()
    out = out.dropna(subset=["event_time_ns"]).reset_index(drop=True)
    return out
