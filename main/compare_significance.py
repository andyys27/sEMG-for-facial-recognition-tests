"""
Compara modelos par a par usando las accuracies POR FOLD 
Usa Wilcoxon + correccion de Holm-Bonferroni por comparaciones multiples

python -m main.compare_significance
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_all_metrics(results_root):
    rows = []
    for metrics_path in sorted(Path(results_root).glob("*_LOSO/*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def fold_acc_by_subject(m):
    return {f["test_subject"]: f["accuracy"] for f in m["fold_results"]}


def holm_bonferroni(pvals):
    # Correccion Holm-Bonferroni
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = max(adj, prev)
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted


def compare_scenario(rows, scenario):
    models = [m for m in rows if m["scenario"] == scenario]
    if len(models) < 2:
        print(f"  (menos de 2 modelos en '{scenario}', nada que comparar)")
        return None

    n_folds = len(models[0]["fold_results"])
    if n_folds < 6:
        print(f"  ⚠ Solo {n_folds} folds en '{scenario}'. Con menos de ~6 pares, "
              f"un Wilcoxon de dos colas no puede llegar a p<0.05 aunque un "
              f"modelo sea claramente mejor. Trata los p-values de abajo como "
              f"orientativos, no como evidencia formal, hasta que agregues "
              f"mas sujetos.")

    records = []
    pvals = []
    for m1, m2 in combinations(models, 2):
        acc1 = fold_acc_by_subject(m1)
        acc2 = fold_acc_by_subject(m2)
        common = sorted(set(acc1) & set(acc2))
        x = np.array([acc1[s] for s in common])
        y = np.array([acc2[s] for s in common])
        diff = x - y

        if len(common) < 2 or np.allclose(diff, 0):
            p = np.nan
        else:
            try:
                _, p = wilcoxon(x, y)
            except ValueError:
                p = np.nan

        records.append({
            "Modelo A": m1["model"], "Modelo B": m2["model"],
            "Acc A (media folds)": float(np.mean(x)), "Acc B (media folds)": float(np.mean(y)),
            "Diferencia (A-B)": float(np.mean(diff)), "n_folds_comparados": len(common),
            "p_value": p,
        })
        pvals.append(p if not np.isnan(p) else 1.0)

    df = pd.DataFrame(records)
    df["p_value_holm"] = holm_bonferroni(df["p_value"].fillna(1.0).values)
    df["significativo (holm<0.05)"] = df["p_value_holm"] < 0.05
    return df.sort_values("Diferencia (A-B)", ascending=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    results_root = root / "Results"
    output_dir = results_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_metrics(results_root)
    if not rows:
        print(f"⚠ No se encontro ningun metrics.json bajo {results_root}. "
              f"Corre primero: python -m main.train_models  y  python -m main.train_cnn")
    else:
        for scenario in sorted(set(m["scenario"] for m in rows)):
            print(f"\n{'=' * 70}\nComparacion pareada - {scenario}\n{'=' * 70}")
            df = compare_scenario(rows, scenario)
            if df is not None:
                print(df.to_string(index=False))
                out_csv = output_dir / f"significance_{scenario}.csv"
                df.to_csv(out_csv, index=False)
                print(f"-> Guardado: {out_csv}")