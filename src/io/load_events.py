"""Lectura de Unity_events.csv (fuente de verdad del protocolo)."""
from pathlib import Path
import pandas as pd
from ._time import timestamp_to_ns

EXPECTED = ["Timestamp", "t_ns", "seq", "label", "detail"]


def parse_detail(detail) -> dict[str, str]:
    """'trial=0;rep=0;gesture=Sonrisa' -> {'trial': '0', ...}."""
    if not isinstance(detail, str):
        return {}
    out = {}
    for part in detail.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_events(path: str | Path) -> pd.DataFrame:
    """Carga Unity_events.csv.

    - `t_ns`: tiempo del evento ya alineado al reloj del PC (unity+offset). Es el
      que se usará para etiquetar.
    - `ts_ns`: el mismo instante reconstruido desde la columna Timestamp (control).
    - `label` se renombra a `event`; `detail` se expande en
      trial / rep / gesture_idx / gesture / reason.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}")
    df = df.sort_values("seq").reset_index(drop=True)
    df["t_ns"] = df["t_ns"].astype("int64")
    df["ts_ns"] = timestamp_to_ns(df["Timestamp"])
    df = df.rename(columns={"label": "event"})

    kv = df["detail"].map(parse_detail)
    for k in ("trial", "rep", "gesture_idx"):
        df[k] = pd.to_numeric(kv.map(lambda d, k=k: d.get(k)), errors="coerce").astype("Int64")
    for k in ("gesture", "reason"):
        df[k] = kv.map(lambda d, k=k: d.get(k))
    return df