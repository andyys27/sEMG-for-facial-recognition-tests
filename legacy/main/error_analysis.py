"""
Analisis de errores a partir de las matrices de confusion guardadas en
metrics.json:
  1. Que par (clase_real -> clase_predicha) se confunde mas seguido, por
     modelo+escenario.
  2. Que clase es sistematicamente dificil para TODOS los modelos (problema 
     de datos/etiquetado/electrodos, no de un modelo puntual) vs. una clase 
     que solo un modelo especifico falla (problema del modelo)

python -m main.error_analysis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_metrics(results_root):
    rows = []
    for metrics_path in sorted(Path(results_root).glob("*_LOSO/*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def top_confusions(m, top_n=3):
    if "confusion_matrix" not in m:
        return None
    cm = np.array(m["confusion_matrix"])
    labels = m["confusion_matrix_labels"]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    rate = cm / row_sums  # normalizado por clase real (fila)

    pairs = []
    for i, real in enumerate(labels):
        for j, pred in enumerate(labels):
            if i == j:
                continue
            pairs.append((real, pred, rate[i, j]))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return [p for p in pairs[:top_n] if p[2] > 0]


def per_class_difficulty(rows, scenario):
    filtered = [m for m in rows if m["scenario"] == scenario]
    if not filtered:
        return None
    classes = sorted(filtered[0]["per_class_f1"].keys())
    data = {c: [m["per_class_f1"][c] for m in filtered] for c in classes}
    df = pd.DataFrame({
        "Clase": classes,
        "F1 promedio (todos los modelos)": [np.mean(data[c]) for c in classes],
        "F1 min": [np.min(data[c]) for c in classes],
        "Modelo con F1 min": [filtered[int(np.argmin(data[c]))]["model"] for c in classes],
        "F1 max": [np.max(data[c]) for c in classes],
        "Modelo con F1 max": [filtered[int(np.argmax(data[c]))]["model"] for c in classes],
    }).sort_values("F1 promedio (todos los modelos)")
    return df


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
            print(f"\n{'=' * 70}\nClases mas dificiles (todos los modelos) - {scenario}\n{'=' * 70}")
            df = per_class_difficulty(rows, scenario)
            print(df.to_string(index=False))
            out_csv = output_dir / f"per_class_difficulty_{scenario}.csv"
            df.to_csv(out_csv, index=False)
            print(f"-> Guardado: {out_csv}")

        print(f"\n{'=' * 70}\nConfusiones mas frecuentes por modelo\n{'=' * 70}")
        for m in rows:
            confusions = top_confusions(m)
            print(f"\n{m['model']} ({m['scenario']}):")
            if confusions is None:
                print("  (metrics.json generado antes del fix de confusion_matrix -> "
                      "vuelve a correr train_models.py / train_cnn.py)")
                continue
            if not confusions:
                print("  (sin confusiones relevantes)")
            for real, pred, rate in confusions:
                print(f"  {real} -> confundido con {pred} en el {rate * 100:.1f}% de los casos")