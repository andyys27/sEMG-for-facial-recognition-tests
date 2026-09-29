from __future__ import annotations

import pandas as pd


def sliding_windows(df: pd.DataFrame, window_size_sec: float = 0.5, overlap_sec: float = 0.25, time_col: str = "time_s") -> pd.DataFrame:
    """Genera ventanas deslizantes a partir de una serie temporal."""
    if df.empty:
        return df.copy()

    step = window_size_sec - overlap_sec
    if step <= 0:
        raise ValueError("El solapamiento debe ser menor que el tamaño de ventana.")

    windows = []
    for start in df[time_col].drop_duplicates().sort_values():
        end = start + window_size_sec
        window_df = df[(df[time_col] >= start) & (df[time_col] < end)].copy()
        if len(window_df) == 0:
            continue
        window_df["window_start_s"] = start
        window_df["window_end_s"] = end
        windows.append(window_df)

    if not windows:
        return pd.DataFrame()

    return pd.concat(windows, ignore_index=True)
