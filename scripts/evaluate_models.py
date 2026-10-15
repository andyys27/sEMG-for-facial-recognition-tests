"""Compara los modelos ya entrenados y diagnostica la generalización.

    python -m scripts.evaluate_models
    python -m scripts.evaluate_models --diagnose rf knn
    python -m scripts.evaluate_models --no-diagnose
"""
import argparse

from src.dataset import load_dataset
from src.evaluation.compare_models import write_comparison
from src.evaluation.diagnose_generalization import diagnose_generalization, write_diagnosis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", nargs="*", default=["rf", "svm", "knn"])
    ap.add_argument("--no-diagnose", action="store_true")
    a = ap.parse_args()

    c = write_comparison()
    print(c["table"].round(3).to_string(index=False))
    print("\nF1 por clase:\n" + c["per_class"].round(2).to_string())
    print(f"\nReporte: {c['summary']}")

    if not a.no_diagnose and a.diagnose:
        df, _ = load_dataset(with_raw=False)
        strategy = "loso" if df.subject.nunique() >= 2 else "leave_one_rep_out"
        t = diagnose_generalization(df, a.diagnose)
        print("\nHonesta vs K-Fold con fuga:\n" + t.round(3).to_string(index=False))
        print(f"Reporte: {write_diagnosis(t, strategy)}")


if __name__ == "__main__":
    main()