"""Métricas de clasificación: por ventana, por fold y por segmento (voto mayoritario)."""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from ..config import load_yaml

SEG_KEYS = ["subject", "session", "segment_id"]
SCALAR_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]


def class_info(labels_cfg: dict | None = None, classes_cfg: dict | None = None):
    """(ids, nombres) de las clases usadas, ordenadas por id."""
    ids = (labels_cfg or load_yaml("experiment/labels.yaml"))["classes"]
    use = (classes_cfg or load_yaml("dataset/classes.yaml"))["use_classes"]
    pairs = sorted((ids[n], n) for n in use)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def classification_metrics(y_true, y_pred, class_ids, names, with_cm: bool = True) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0)
        out = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=class_ids, average="weighted", zero_division=0)),
            "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                              "support": int(s[i])} for i, n in enumerate(names)},
        }
    if with_cm:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_ids).tolist()
    return out


def segment_vote(pred: pd.DataFrame) -> pd.DataFrame:
    """Una predicción por segmento: la clase más votada entre sus ventanas.
    Es la métrica relevante para una aplicación que decide por gesto completo."""
    g = pred.groupby(SEG_KEYS, sort=False)
    return g.agg(y_true=("y_true", "first"),
                 y_pred=("y_pred", lambda s: s.mode().iloc[0]),
                 n_windows=("y_pred", "size")).reset_index()


def summarize(pred: pd.DataFrame, class_ids, names) -> dict:
    """pred: columnas fold, subject, session, segment_id, y_true, y_pred (una fila por ventana de test)."""
    pooled = classification_metrics(pred["y_true"], pred["y_pred"], class_ids, names)
    per_fold = {f: classification_metrics(d["y_true"], d["y_pred"], class_ids, names, with_cm=False)
                for f, d in pred.groupby("fold", sort=False)}
    stats = {}
    for k in SCALAR_METRICS:
        v = [m[k] for m in per_fold.values()]
        stats[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    seg = segment_vote(pred)
    return {"n_folds": len(per_fold), "pooled": pooled, "per_fold": per_fold, "fold_stats": stats,
            "segment_vote": classification_metrics(seg["y_true"], seg["y_pred"], class_ids, names)}