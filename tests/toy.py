"""Datasets sintéticos mínimos para probar modelos y evaluación sin datos reales."""
import numpy as np
import pandas as pd

CLASSES = {"Reposo": 0, "Sonrisa": 1, "Triste": 2, "Disgusto": 3, "Sorprendido": 4}


def make_toy_df(separable: bool = True, reps: int = 5, win_per_seg: int = 6, n_feat: int = 6, seed: int = 0):
    """Un sujeto, `reps` repeticiones (+ reposo inicial con rep=-1).

    separable=True : la media de las features depende de la clase (se puede generalizar).
    separable=False: las features dependen del SEGMENTO y no de la clase (solo se puede
                     memorizar el segmento; un K-Fold por ventanas lo "aprende", uno honesto no).
    """
    rng = np.random.default_rng(seed)
    gestures = [n for n in CLASSES if n != "Reposo"]
    rows, sid = [], 0
    for rep in [-1, *range(reps)]:
        labels = ["Reposo"] if rep == -1 else [l for g in gestures for l in (g, "Reposo")]
        for k, lab in enumerate(labels):
            seg_sig = rng.normal(0, 3, n_feat)
            mean = np.eye(n_feat)[CLASSES[lab]] * 5 if separable else seg_sig
            for w in range(win_per_seg):
                rows.append({"subject": "s1", "session": "session_01", "segment_id": sid, "window_idx": w,
                             "trial": -1 if rep == -1 else rep * 4 + k // 2, "rep": rep, "label": lab,
                             "label_id": CLASSES[lab], **{f"f{j}": v for j, v in enumerate(mean + rng.normal(0, 1, n_feat))}})
            sid += 1
    return pd.DataFrame(rows)


def make_toy_raw(df: pd.DataFrame, seed: int = 0) -> dict:
    """Ventanas crudas (n, 200, 4) cuya amplitud crece x3 por clase."""
    rng = np.random.default_rng(seed)
    amp = (3.0 ** df["label_id"].to_numpy())[:, None, None] * 1e-5
    return {"windows": (rng.normal(0, 1, (len(df), 200, 4)) * amp).astype("float32")}


def small_cfg() -> dict:
    from src.models import load_models_cfg
    cfg = load_models_cfg()
    cfg["models"]["rf"] = {"n_estimators": 40, "n_jobs": 1}
    cfg["models"]["mlp"] = {"hidden_layer_sizes": [16], "max_iter": 200}
    cfg["cnn"] = {"filters": [8, 8], "epochs": 20, "batch_size": 16, "patience": 20, "val_frac": 0.2}
    return cfg