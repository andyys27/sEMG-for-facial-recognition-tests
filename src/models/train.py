"""Entrenamiento y evaluación con validación cruzada por grupos (sin fuga)."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from ..dataset.build_dataset import feature_columns
from ..dataset.split_dataset import get_splits
from ..evaluation.confusion_matrix import plot_confusion_matrix
from ..evaluation.metrics import class_info, summarize
from .baseline_models import build_model, load_models_cfg
from .common import models_root, prediction_frame, undersample_indices
from .save_model import save_model


def cross_validate(df: pd.DataFrame, model_name: str, folds=None, models_cfg: dict | None = None,
                   seed: int | None = None) -> pd.DataFrame:
    """Entrena un modelo por fold y devuelve las predicciones de test (una fila por ventana).
    Escalado, balanceo y ajuste usan SOLO las filas de entrenamiento de cada fold."""
    cfg = models_cfg or load_models_cfg()
    seed = cfg.get("seed", 42) if seed is None else seed
    X = df[feature_columns(df)].to_numpy(dtype="float64")
    y = df["label_id"].to_numpy()
    folds = list(get_splits(df)) if folds is None else list(folds)

    parts = []
    for f in folds:
        tr = f.train
        if cfg.get("balance_train", True):
            tr = tr[undersample_indices(y[tr], seed)]
        model = build_model(model_name, cfg, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # LDA avisa de features colineales (p. ej. MAV/IEMG)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[f.test])
        parts.append(prediction_frame(df, f.test, pred, f.name))
    return pd.concat(parts, ignore_index=True)


def fit_final(df: pd.DataFrame, model_name: str, models_cfg: dict | None = None, seed: int | None = None):
    """Modelo entrenado con TODAS las ventanas (para usarlo después, no para evaluarlo)."""
    cfg = models_cfg or load_models_cfg()
    seed = cfg.get("seed", 42) if seed is None else seed
    X = df[feature_columns(df)].to_numpy(dtype="float64")
    y = df["label_id"].to_numpy()
    idx = undersample_indices(y, seed) if cfg.get("balance_train", True) else np.arange(len(y))
    model = build_model(model_name, cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[idx], y[idx])
    return model


def run_experiment(df: pd.DataFrame, model_name: str, raw: dict | None = None, folds=None,
                   models_cfg: dict | None = None, out_root: str | Path | None = None,
                   save_final: bool = True) -> dict:
    """Evalúa un modelo y escribe en data/analysis/models/<modelo>/:
    metrics.json, predictions.csv, confusion_matrix.png y (tabulares) model.joblib."""
    cfg = models_cfg or load_models_cfg()
    seed = cfg.get("seed", 42)
    folds = list(get_splits(df)) if folds is None else list(folds)

    if model_name == "cnn":
        from .cnn_model import cross_validate_cnn          # import perezoso: tensorflow es opcional
        if raw is None:
            raise ValueError("La CNN necesita las ventanas crudas (raw_windows.npz)")
        pred = cross_validate_cnn(df, raw["windows"], folds, cfg, seed)
    else:
        pred = cross_validate(df, model_name, folds, cfg, seed)

    ids, names = class_info()
    summary = summarize(pred, ids, names)
    params = cfg.get("cnn") if model_name == "cnn" else (cfg.get("models") or {}).get(model_name)
    meta = {"model": model_name, "params": params, "balance_train": cfg.get("balance_train", True),
            "seed": seed, "folds": [f.name for f in folds], "n_windows": int(len(df)),
            "n_features": len(feature_columns(df)), "classes": names}

    out = Path(out_root) if out_root else models_root()
    out = out / model_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, **summary}, indent=2), encoding="utf-8")
    id2n = dict(zip(ids, names))
    pred.assign(y_true_name=pred["y_true"].map(id2n), y_pred_name=pred["y_pred"].map(id2n)) \
        .to_csv(out / "predictions.csv", index=False)
    p = summary["pooled"]
    plot_confusion_matrix(p["confusion_matrix"], names,
                          f"{model_name} | macro-F1 {p['macro_f1']:.2f} | acc {p['accuracy']:.2f} "
                          f"| {len(folds)} folds", out / "confusion_matrix.png")
    if save_final and model_name != "cnn":
        save_model(fit_final(df, model_name, cfg, seed), out / "model.joblib", meta)
    return {"summary": summary, "pred": pred, "dir": out, "meta": meta}