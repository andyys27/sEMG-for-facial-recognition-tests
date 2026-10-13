"""Features de dominio del tiempo (Hudgins 1993; Phinyomark 2012), vectorizadas."""
import numpy as np


def temporal_features(W: np.ndarray, channels: list[str], thr: float) -> dict[str, np.ndarray]:
    """W: (n_windows, L, n_channels) -> {'<canal>_<feature>': (n_windows,)}.

    MAV, RMS, WL, VAR, IEMG, ZC y SSC. ZC y SSC usan un umbral de amplitud `thr`
    (en las unidades de la señal) para ignorar ruido.
    """
    x = W.astype("float64")
    d = np.diff(x, axis=1)
    a, b = x[:, 1:-1] - x[:, :-2], x[:, 1:-1] - x[:, 2:]
    feats = {
        "MAV": np.abs(x).mean(axis=1),
        "RMS": np.sqrt((x ** 2).mean(axis=1)),
        "WL": np.abs(d).sum(axis=1),
        "VAR": x.var(axis=1, ddof=1),
        "IEMG": np.abs(x).sum(axis=1),
        "ZC": ((x[:, :-1] * x[:, 1:] < 0) & (np.abs(d) >= thr)).sum(axis=1),
        "SSC": ((a * b > 0) & (np.abs(a) >= thr) & (np.abs(b) >= thr)).sum(axis=1),
    }
    return {f"{ch}_{name}": v[:, j] for name, v in feats.items() for j, ch in enumerate(channels)}