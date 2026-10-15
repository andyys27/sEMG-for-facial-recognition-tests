"""Entrena y evalúa modelos con validación cruzada por grupos (sin fuga).

    python -m scripts.train_models                    # dummy rf svm knn lda mlp
    python -m scripts.train_models --models rf svm
    python -m scripts.train_models --cnn              # agrega la CNN (requiere tensorflow)
    python -m scripts.train_models --list
"""
import argparse

import pandas as pd

from src.dataset import get_splits, load_dataset
from src.models import MODEL_NAMES, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("Modelos:", ", ".join(MODEL_NAMES + ["cnn"]))
        return

    names = list(dict.fromkeys(a.models + (["cnn"] if a.cnn else [])))
    df, raw = load_dataset(with_raw="cnn" in names)
    folds = list(get_splits(df))
    print(f"{len(df)} ventanas | {df.subject.nunique()} sujeto(s) | {len(folds)} folds: "
          f"{[f.name for f in folds]}\n")

    rows = []
    for name in names:
        try:
            r = run_experiment(df, name, raw=raw, folds=folds)
        except ImportError as e:
            print(f"[{name}] omitido: {e}")
            continue
        s = r["summary"]
        rows.append({"model": name, "macro_f1": s["pooled"]["macro_f1"],
                     "fold_std": s["fold_stats"]["macro_f1"]["std"],
                     "bal_acc": s["pooled"]["balanced_accuracy"], "acc": s["pooled"]["accuracy"],
                     "seg_macro_f1": s["segment_vote"]["macro_f1"]})
        print(f"[{name}] macro-F1={rows[-1]['macro_f1']:.3f}  -> {r['dir']}")
    print("\n" + pd.DataFrame(rows).sort_values("macro_f1", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()