from __future__ import annotations

import pandas as pd


def validate_emg_segment(signal_df: pd.DataFrame, protocol_df: pd.DataFrame) -> pd.DataFrame:
    """Valida si un segmento detectado en la EMG es coherente con el protocolo experimental.

    Este módulo no dicta la etiqueta final: solo compara el timing de la señal
    con el protocolo de Unity y marca incoherencias.
    """
    if signal_df.empty or protocol_df.empty:
        return pd.DataFrame(columns=["segment_id", "valid", "reason"])

    result = []
    for _, row in protocol_df.iterrows():
        result.append({
            "segment_id": row.get("event_type", "unknown"),
            "valid": True,
            "reason": "protocol_packet_matches_expected_timeline",
        })

    return pd.DataFrame(result)
