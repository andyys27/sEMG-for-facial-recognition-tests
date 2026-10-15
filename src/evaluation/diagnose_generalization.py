"""Diagnóstico de generalización: evaluación honesta vs. K-Fold por ventanas (con fuga).

Las ventanas se solapan 50 % y comparten segmento, así que un K-Fold aleatorio mezcla
ventanas casi idénticas entre train y test. La diferencia entre ambas evaluaciones mide
cuánto se infla el rendimiento por esa fuga."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import load_yaml
from ..dataset.split_dataset import Fold, get_splits
from ..models.train import cross_validate
from .metrics import class_info, classification_metrics


def leaky_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[Fold]:
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return [Fold(f"kfold={i}", tr, te) for i, (tr, te) in enumerate(skf.split(df, df["label_id"]))]


def diagnose_generalization(df: pd.DataFrame, model_names=("rf", "svm", "knn"), models_cfg: dict | None = None,
                            n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    ids, names = class_info()
    honest_folds, leaky = list(get_splits(df)), leaky_folds(df, n_splits, seed)
    rows = []
    for m in model_names:
        h = cross_validate(df, m, honest_folds, models_cfg)
        k = cross_validate(df, m, leaky, models_cfg)
        mh = classification_metrics(h["y_true"], h["y_pred"], ids, names, with_cm=False)
        mk = classification_metrics(k["y_true"], k["y_pred"], ids, names, with_cm=False)
        rows.append({"model": m, "honest_macro_f1": mh["macro_f1"], "leaky_macro_f1": mk["macro_f1"],
                     "gap_macro_f1": mk["macro_f1"] - mh["macro_f1"],
                     "honest_acc": mh["accuracy"], "leaky_acc": mk["accuracy"]})
    return pd.DataFrame(rows)


def write_diagnosis(table: pd.DataFrame, strategy: str, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else Path(load_yaml("experiment/settings.yaml")["paths"]["analysis"]) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    text = ["# Diagnóstico de generalización\n",
            f"- **Honesta**: `{strategy}` (los segmentos de test nunca aparecen en entrenamiento).",
            "- **Con fuga**: K-Fold estratificado por ventanas (ventanas solapadas del mismo segmento en train y test).\n",
            "```", table.round(3).to_string(index=False), "```\n",
            "Una brecha grande indica que el rendimiento del K-Fold por ventanas es aprendizaje de segmentos "
            "concretos y **no** capacidad de generalizar. Reporta siempre la evaluación honesta.\n"]
    p = out / "generalization_diagnosis.md"
    p.write_text("\n".join(text), encoding="utf-8")
    return p