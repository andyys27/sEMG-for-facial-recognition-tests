"""
Genera y guarda las graficas de evaluacion del modelo Random Forest (LOSO):
  - Matriz de confusion (agregada, todos los folds)
  - Importancia de features (top N, promedio entre folds)
  - Precision / Recall / F1 por clase (agregado)
  - Accuracy por fold (por sujeto excluido)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .model import run_loso


def save_confusion_matrix(cm, labels, save_path, title="Matriz de confusion"):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_feature_importance(imp_df, save_path, top_n=15, title="Importancia de features"):
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("Importancia promedio (folds LOSO)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_class_metrics(report_dict, save_path, title="Precision / Recall / F1 por clase"):
    classes = [k for k in report_dict.keys()
               if k not in ("accuracy", "macro avg", "weighted avg")]
    precision = [report_dict[c]["precision"] for c in classes]
    recall = [report_dict[c]["recall"] for c in classes]
    f1 = [report_dict[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_fold_accuracy(fold_results, save_path, title="Accuracy por fold (sujeto excluido)"):
    subjects = [f["test_subject"] for f in fold_results]
    accs = [f["accuracy"] for f in fold_results]
    mean_acc = float(np.mean(accs))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(subjects, accs, color="#55A868")
    ax.axhline(mean_acc, color="crimson", linestyle="--", linewidth=1,
               label=f"Promedio = {mean_acc:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_all_plots(df, feature_cols, output_dir, only_high_confidence=False,
                        n_estimators=300, random_state=42, top_n_features=15):
    # Corre LOSO una sola vez y guarda las 4 graficas dentro de output_dir
    # Devuelve el dict de resultados de run_loso 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_loso(df, feature_cols, only_high_confidence=only_high_confidence,
                        n_estimators=n_estimators, random_state=random_state,
                        return_results=True)

    if results is None:
        print(f"  (sin resultados: no se generaron graficas en {output_dir})")
        return None

    save_confusion_matrix(results["cm"], results["labels"], output_dir / "confusion_matrix.png")
    save_feature_importance(results["imp_df"], output_dir / "feature_importance.png", top_n=top_n_features)
    save_class_metrics(results["report_dict"], output_dir / "class_metrics.png")
    save_fold_accuracy(results["fold_results"], output_dir / "fold_accuracy.png")

    print(f"  -> Graficas guardadas en: {output_dir}")
    return results