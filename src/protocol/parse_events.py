"""Convierte los eventos de Unity en marcas tipadas según config/protocol/events.yaml."""
import warnings
import pandas as pd
from ..config import load_yaml
from ..io.load_events import parse_detail


def parse_events(events: pd.DataFrame, events_cfg: dict | None = None) -> pd.DataFrame:
    """Devuelve una marca por evento: t_ns, event, phase, trial, rep, gesture_idx,
    gesture, reason. `phase` es None para eventos estructurales."""
    mapping = (events_cfg or load_yaml("protocol/events.yaml"))["events"]
    unknown = sorted(set(events["event"]) - set(mapping))
    if unknown:
        warnings.warn(f"Eventos de Unity no definidos en events.yaml: {unknown}")
    m = events[["t_ns", "event", "trial", "rep", "gesture_idx", "gesture", "reason"]].copy()
    m["phase"] = m["event"].map(mapping)
    for c in ("trial", "rep", "gesture_idx"):
        m[c] = m[c].fillna(-1).astype("int64")
    return m.sort_values("t_ns", kind="stable").reset_index(drop=True)


def protocol_params(events: pd.DataFrame) -> dict:
    """Parámetros del protocolo desde `sequence_start` (reps, gestures, rest_s, hold_s...)."""
    row = events.loc[events["event"] == "sequence_start"]
    if row.empty:
        return {}
    out = {}
    for k, v in parse_detail(row.iloc[0]["detail"]).items():
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out