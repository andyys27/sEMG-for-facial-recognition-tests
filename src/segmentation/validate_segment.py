"""Validación de segmentos (duración, muestras faltantes, NaN, canal plano)."""
import numpy as np
from .segment_signal import Segment


def validate_segment(seg: Segment, fs_hz: float, min_duration_s: float,
                     min_coverage: float = 0.95) -> tuple[bool, str]:
    if seg.duration_s < min_duration_s:
        return False, "too_short"
    if len(seg.data) < min_coverage * seg.duration_s * fs_hz:
        return False, "missing_samples"
    if np.isnan(seg.data).any():
        return False, "nan"
    if (seg.data.std(axis=0) == 0).any():
        return False, "flatline"
    return True, "ok"


def validate_segments(segs: list[Segment], fs_hz: float, min_duration_s: float) -> list[Segment]:
    """Marca `valid` y `reason` en cada segmento (no los elimina)."""
    for s in segs:
        s.valid, s.reason = validate_segment(s, fs_hz, min_duration_s)
    return segs