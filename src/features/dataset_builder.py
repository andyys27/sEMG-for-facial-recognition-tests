from __future__ import annotations

import pandas as pd


def build_feature_dataset(labels_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """Une etiquetas del protocolo con la señal para crear el dataset ML.

    En una implementación completa aquí se introducirían features por ventana,
    estadísticos por canal y validaciones adicionales.
    """
    merged = labels_df.merge(signal_df, how="left", left_on="event_time_ns", right_on="time_ns")
    return merged.reset_index(drop=True)
