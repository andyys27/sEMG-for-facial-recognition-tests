"""
Recorre Results/<Modelo>_LOSO/<escenario>/metrics.json y arma:
  - una tabla comparativa entre todos los modelos entrenados
  - grafica de accuracy por modelo
  - grafica de macro F1 por modelo
  - heatmap de F1 por clase y modelo

python -m main.compare_models
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_metrics(results_root):
    rows = []
    for metrics_path in sorted(Path(results_root).glob("*_LOSO/*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def build_summary_table(rows):
    records = [{
        "Modelo": m["model"],
        "Escenario": m["scenario"],
        "Accuracy (agregado)": m["overall_accuracy"],
        "Accuracy (media folds)": m["fold_accuracy_mean"],
        "Accuracy (std folds)": m["fold_accuracy_std"],
        "Macro F1": m["macro_f1"],
        "Weighted F1": m["weighted_f1"],
    } for m in rows]
    return pd.DataFrame(records).sort_values(["Escenario", "Macro F1"], ascending=[True, False])


def plot_accuracy_comparison(df, save_path):
    scenarios = df["Escenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["Escenario"] == scenario].sort_values("Accuracy (agregado)", ascending=False)
        bars = ax.bar(sub["Modelo"], sub["Accuracy (agregado)"], yerr=sub["Accuracy (std folds)"], capsize=4, color="#4C72B0")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Accuracy - {scenario}")
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, sub["Accuracy (agregado)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_macro_f1_comparison(df, save_path):
    scenarios = df["Escenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["Escenario"] == scenario].sort_values("Macro F1", ascending=False)
        bars = ax.bar(sub["Modelo"], sub["Macro F1"], color="#55A868")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Macro F1 - {scenario}")
        ax.set_ylabel("Macro F1")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, sub["Macro F1"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                    ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1_heatmap(rows, scenario, save_path):
    filtered = [m for m in rows if m["scenario"] == scenario]
    if not filtered:
        return
    classes = sorted(filtered[0]["per_class_f1"].keys())
    models = [m["model"] for m in filtered]
    data = np.array([[m["per_class_f1"][c] for c in classes] for m in filtered])

    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 3, 0.6 * len(models) + 2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=20)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(f"F1 por clase y modelo - {scenario}")

    for i in range(len(models)):
        for j in range(len(classes)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="white" if data[i, j] > 0.5 else "black", fontsize=9)

    fig.colorbar(im, ax=ax, label="F1-score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    results_root = root / "Results"
    output_dir = results_root / "comparison"

    rows = load_all_metrics(results_root)
    if not rows:
        print(f"⚠ No se encontro ningun metrics.json bajo {results_root}. "
              f"Corre primero: python -m main.train_models  y  python -m main.train_cnn")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        df = build_summary_table(rows)
        print(df.to_string(index=False))

        csv_path = output_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n-> Tabla guardada: {csv_path}")

        plot_accuracy_comparison(df, output_dir / "accuracy_comparison.png")
        plot_macro_f1_comparison(df, output_dir / "macro_f1_comparison.png")

        for scenario in df["Escenario"].unique():
            plot_per_class_f1_heatmap(rows, scenario, output_dir / f"per_class_f1_{scenario}.png")

        print(f"-> Graficas guardadas en: {output_dir}")