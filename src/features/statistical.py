"""Features estadísticas de la envolvente RMS (cruda y normalizada por z-score)."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def envelope(W: np.ndarray, fs_hz: float, env_ms: float) -> np.ndarray:
    """Envolvente RMS móvil dentro de cada ventana: (n, L, C)."""
    k = max(1, int(round(env_ms * fs_hz / 1000)))
    return np.sqrt(uniform_filter1d(W.astype("float64") ** 2, size=k, axis=1, mode="nearest"))


def envelope_baseline(W: np.ndarray, fs_hz: float, env_ms: float) -> dict[str, np.ndarray]:
    """Media y desviación de la envolvente por canal sobre TODAS las ventanas de una
    sesión. No usa etiquetas: es una normalización por sujeto/sesión."""
    env = envelope(W, fs_hz, env_ms)
    return {"mu": env.mean(axis=(0, 1)), "sd": env.std(axis=(0, 1)) + 1e-30}


def statistical_features(W: np.ndarray, channels: list[str], fs_hz: float, env_ms: float,
                         baseline: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    env = envelope(W, fs_hz, env_ms)
    z = (env - baseline["mu"]) / baseline["sd"]
    feats = {"env_mean": env.mean(axis=1), "env_max": env.max(axis=1), "env_std": env.std(axis=1),
             "z_mean": z.mean(axis=1), "z_max": z.max(axis=1)}
    return {f"{ch}_{name}": v[:, j] for name, v in feats.items() for j, ch in enumerate(channels)}