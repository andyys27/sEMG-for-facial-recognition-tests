"""Ventaneo con solapamiento y exportación de ventanas."""
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .segment_signal import Segment

META_KEYS = ("segment_id", "label", "label_id", "trial", "rep")


def temporal_windowing(x: np.ndarray, fs_hz: float, window_ms: float, overlap: float) -> np.ndarray:
    """x: (n_samples, n_channels) -> (n_windows, window_len, n_channels)."""
    n = int(round(window_ms * fs_hz / 1000))
    step = max(1, int(round(n * (1 - overlap))))
    if len(x) < n:
        return np.empty((0, n, x.shape[1]), dtype=x.dtype)
    w = sliding_window_view(x, n, axis=0)                  # (len-n+1, ch, n)
    return np.ascontiguousarray(w[::step].transpose(0, 2, 1))


def windows_from_segments(segs: list[Segment], fs_hz: float, window_ms: float,
                          overlap: float) -> dict[str, np.ndarray]:
    """Ventanas de los segmentos válidos; cada ventana hereda label/trial/rep del segmento."""
    chunks, meta = [], {k: [] for k in META_KEYS}
    for s in segs:
        if not s.valid:
            continue
        w = temporal_windowing(s.data, fs_hz, window_ms, overlap)
        if len(w) == 0:
            continue
        chunks.append(w.astype("float32"))
        for k in META_KEYS:
            meta[k].extend([getattr(s, k)] * len(w))
    out = {"windows": np.concatenate(chunks) if chunks else np.empty((0, 0, 0), "float32")}
    out.update({k: np.array(v) for k, v in meta.items()})
    return out


def save_windows(path: str | Path, data: dict, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, **extra)
    return path