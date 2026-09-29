from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io.load_unity_events import load_unity_events
from src.labeling.assign_phases import assign_intervals


def build_labels_from_unity(unity_csv_path: str | Path, subject_id: str | None = None, session_id: str | None = None) -> pd.DataFrame:
    """Construye una tabla de etiquetas de protocolo a partir de events de Unity."""
    events = load_unity_events(unity_csv_path)
    labels = assign_intervals(events)

    labels["subject_id"] = subject_id or "unknown_subject"
    labels["session_id"] = session_id or "unknown_session"
    labels["source"] = "unity"
    labels["confidence"] = "high"

    for col in [
        "trial",
        "rep",
        "gesture_idx",
        "gesture",
        "event_type",
        "phase",
        "label",
        "event_time_ns",
        "start_time_ns",
        "end_time_ns",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "source",
        "confidence",
    ]:
        if col not in labels.columns:
            labels[col] = None

    return labels[[
        "subject_id",
        "session_id",
        "trial",
        "rep",
        "gesture_idx",
        "gesture",
        "event_type",
        "phase",
        "label",
        "event_time_ns",
        "start_time_ns",
        "end_time_ns",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "source",
        "confidence",
    ]].reset_index(drop=True)


def export_labels_csv(labels_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(path, index=False)
    return path
