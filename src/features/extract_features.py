"""Une todos los grupos de features en un vector por ventana."""
from itertools import combinations
import numpy as np
import pandas as pd

from ..config import load_yaml
from .temporal import temporal_features
from .frequency import frequency_features
from .statistical import statistical_features, envelope_baseline


def extract_features(windows: np.ndarray, fs_hz: float, channels: list[str],
                     cfg: dict | None = None, baseline: dict | None = None) -> pd.DataFrame:
    """windows: (n, L, C) -> DataFrame con una fila por ventana.

    Columnas: <canal>_<feature> (temporal, frecuencia, envolvente/z-score) y, para cada
    par de grupos musculares, ratio_<A>_<B> y diff_<A>_<B> sobre el RMS de grupo.
    `baseline` (media/sd de envolvente) se calcula de las propias ventanas si no se da.
    """
    cfg = cfg or load_yaml("dataset/features.yaml")
    env_ms = cfg["envelope_ms"]
    baseline = baseline or envelope_baseline(windows, fs_hz, env_ms)

    feats = {}
    feats.update(temporal_features(windows, channels, cfg["zc_ssc_threshold"]))
    feats.update(frequency_features(windows, channels, fs_hz, cfg["bands_hz"]))
    feats.update(statistical_features(windows, channels, fs_hz, env_ms, baseline))

    grp = {g: np.sqrt(np.mean([feats[f"{c}_RMS"] ** 2 for c in cs], axis=0))
           for g, cs in cfg["channel_groups"].items()}
    for a, b in combinations(grp, 2):
        feats[f"ratio_{a}_{b}"] = grp[a] / (grp[b] + 1e-12)
        feats[f"diff_{a}_{b}"] = grp[a] - grp[b]
    return pd.DataFrame(feats)