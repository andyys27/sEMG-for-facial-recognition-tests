"""Asocia cada intervalo con su etiqueta (emoción) según el protocolo."""
import pandas as pd
from ..config import load_yaml


def map_emotions(intervals: pd.DataFrame, labels_cfg: dict | None = None) -> pd.DataFrame:
    """Añade `label` y `label_id`.

    Gesto -> nombre del gesto (Sonrisa...), Reposo -> 'Reposo', Countdown -> 'Countdown'.
    El reposo posterior a un gesto conserva `gesture` en el detalle de Unity,
    pero su label es siempre 'Reposo'.
    """
    cfg = labels_cfg or load_yaml("experiment/labels.yaml")
    ids = {**cfg["classes"], **cfg["special"]}
    out = intervals.copy()
    out["label"] = out["phase"]
    is_g = out["phase"].eq("Gesto")
    out.loc[is_g, "label"] = out.loc[is_g, "gesture"]
    bad = sorted(set(out["label"]) - set(ids))
    if bad:
        raise ValueError(f"Etiquetas sin id en labels.yaml: {bad}")
    out["label_id"] = out["label"].map(ids).astype("int64")
    return out