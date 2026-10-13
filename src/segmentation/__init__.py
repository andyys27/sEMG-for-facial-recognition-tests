"""Filtrado, recorte y ventaneo. Apoya al protocolo, no lo reemplaza."""
from .preprocess import preprocess_emg
from .segment_signal import Segment, segment_signal
from .validate_segment import validate_segment, validate_segments
from .temporal_windowing import temporal_windowing, windows_from_segments, save_windows

__all__ = ["preprocess_emg", "Segment", "segment_signal", "validate_segment",
           "validate_segments", "temporal_windowing", "windows_from_segments", "save_windows"]