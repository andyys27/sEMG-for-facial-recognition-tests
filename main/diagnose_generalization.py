"""
Diagnostico: LOSO (generalizacion real entre sujetos) vs. un K-Fold
estratificado normal que IGNORA el sujeto 
Sirve para responder una pregunta clave que separa dos causas muy 
distintas de un F1 bajo:

  Si LOSO da F1 ~0 en una clase pero el K-Fold "con fuga" da F1 alto ->
    la senal SI es discriminativa en principio; el problema es que el
    gesto de esa emocion varia demasiado entre tus sujetos actuales
    (solucion: mas sujetos, no otro modelo)

  Si AMBOS dan F1 bajo en la misma clase ->
    el problema no es de generalizacion entre sujetos: es de la
    senal/etiqueta en si (revisa segmentacion, sincronizacion
    Block/Emotion/Label, o si el gesto se registro bien en esos canales)

python -m main.diagnose_generalization
python -m main.diagnose_generalization --model svm
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from .emg_ml.model import load_dataset, run_loso
from .emg_ml.models import build_model, MODEL_NAMES


def class_balance_report(df):
    table = df.groupby(["Subject", "Label"]).size().unstack(fill_value=0)
    print("Ventanas por clase y sujeto:")
    print(table.to_string())
    print()
    print("Total de ventanas por clase (todos los sujetos):")
    print(df["Label"].value_counts().to_string())
    return table


def kfold_with_leakage(df, feature_cols, model_name="rf", n_splits=5, random_state=42):
    # K-Fold estratificado normal, SIN agrupar por sujeto
    X = df[feature_cols].values
    y = df["Label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        clf = build_model(model_name, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(y_pred)

    return classification_report(all_true, all_pred, zero_division=0, output_dict=True)


def compare_loso_vs_leakage(df, feature_cols, model_name, model_label):
    print(f"\n{'=' * 70}\n{model_label}\n{'=' * 70}")

    print("\n-- LOSO (generalizacion real entre sujetos) --")
    loso_results = run_loso(df, feature_cols, model_name=model_name, return_results=True)
    loso_report = loso_results["report_dict"] if loso_results else None

    print("\n-- K-Fold con fuga de sujeto (SOLO diagnostico, no es metrica valida) --")
    leak_report = kfold_with_leakage(df, feature_cols, model_name=model_name)

    classes = sorted(df["Label"].unique())
    rows = []
    for c in classes:
        loso_f1 = loso_report[c]["f1-score"] if loso_report and c in loso_report else np.nan
        leak_f1 = leak_report[c]["f1-score"] if c in leak_report else np.nan
        gap = (leak_f1 - loso_f1) if not (np.isnan(loso_f1) or np.isnan(leak_f1)) else np.nan
        rows.append({
            "Clase": c,
            "F1 LOSO (real)": loso_f1,
            "F1 K-Fold con fuga (optimista)": leak_f1,
            "Brecha (fuga - LOSO)": gap,
        })

    summary = pd.DataFrame(rows).sort_values("Brecha (fuga - LOSO)", ascending=False)
    print("\nResumen:")
    print(summary.to_string(index=False))

    print("\nInterpretacion:")
    for _, r in summary.iterrows():
        leak_f1, gap = r["F1 K-Fold con fuga (optimista)"], r["Brecha (fuga - LOSO)"]
        if np.isnan(gap):
            continue
        if leak_f1 < 0.15:
            print(f"  {r['Clase']}: F1 bajo en AMBOS -> sospecha de datos/etiquetado, "
                  f"no de generalizacion entre sujetos.")
        elif gap > 0.3:
            print(f"  {r['Clase']}: brecha grande -> la senal SI es discriminativa, "
                  f"el problema es que no generaliza entre tus sujetos actuales "
                  f"(necesitas mas sujetos).")
        else:
            print(f"  {r['Clase']}: brecha moderada/pequena.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rf", choices=list(MODEL_NAMES), help="Modelo a usar para el diagnostico (default: rf, es el mas rapido)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"

    df, feature_cols = load_dataset(dataset_path)
    print(f"Dataset: {len(df)} ventanas, sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}\n")

    class_balance_report(df)
    compare_loso_vs_leakage(df, feature_cols, model_name=args.model, model_label=MODEL_NAMES[args.model])