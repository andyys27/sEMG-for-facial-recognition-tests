"""Features de dominio de la frecuencia, vectorizadas."""
import numpy as np


def frequency_features(W: np.ndarray, channels: list[str], fs_hz: float,
                       bands_hz: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """MNF (frecuencia media), MDF (mediana) y potencia relativa por banda."""
    x = W.astype("float64") * np.hanning(W.shape[1])[None, :, None]
    p = np.abs(np.fft.rfft(x, axis=1)) ** 2                  # (n, F, C)
    f = np.fft.rfftfreq(W.shape[1], 1 / fs_hz)
    tot = p.sum(axis=1) + 1e-30

    feats = {"MNF": (p * f[None, :, None]).sum(axis=1) / tot}
    cum = np.cumsum(p, axis=1)
    feats["MDF"] = f[(cum >= tot[:, None, :] / 2).argmax(axis=1)]
    for name, (lo, hi) in bands_hz.items():
        m = (f >= lo) & (f < hi)
        feats[f"BP_{name}"] = p[:, m].sum(axis=1) / tot
    return {f"{ch}_{name}": v[:, j] for name, v in feats.items() for j, ch in enumerate(channels)}