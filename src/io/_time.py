"""Conversión de timestamps a ns desde epoch (UTC naive, como los guarda Unity)."""
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")


def timestamp_to_ns(s: pd.Series) -> pd.Series:
    """'2026-09-24 19:44:49.288088200' -> int64 ns. Exacto hasta 1 ns."""
    dt = pd.to_datetime(s, format="ISO8601")
    return ((dt - _EPOCH) // pd.Timedelta(1, "ns")).astype("int64")