"""
Genera y guarda las graficas de evaluacion del modelo Random Forest (LOSO):
  - Matriz de confusion (agregada, todos los folds)
  - Importancia de features (top N, promedio entre folds)
  - Precision / Recall / F1 por clase (agregado)
  - Accuracy por fold (por sujeto excluido)
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
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
    top = imp_df.head(top_n).iloc[::-1]  # invertido: el mas importante queda arriba
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


def save_metrics_json(results, save_path, model_label, scenario):
    # Guarda un resumen numerico de los resultados de un modelo+escenario
    report = results["report_dict"]
    classes = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

    fold_results_clean = [
        {
            "test_subject": str(f["test_subject"]),
            "n_train": int(f["n_train"]),
            "n_test": int(f["n_test"]),
            "accuracy": float(f["accuracy"]),
        }
        for f in results["fold_results"]
    ]
    fold_accs = [f["accuracy"] for f in fold_results_clean]

    metrics = {
        "model": model_label,
        "scenario": scenario,
        "overall_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class_precision": {c: float(report[c]["precision"]) for c in classes},
        "per_class_recall": {c: float(report[c]["recall"]) for c in classes},
        "per_class_f1": {c: float(report[c]["f1-score"]) for c in classes},
        "fold_accuracy_mean": float(np.mean(fold_accs)),
        "fold_accuracy_std": float(np.std(fold_accs)),
        "fold_results": fold_results_clean,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def generate_all_plots(df, feature_cols, output_dir, model_name="rf", model_label=None, only_high_confidence=False, random_state=42, top_n_features=15):
    # Corre LOSO una sola vez y guarda las graficas
    from .models import MODEL_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = model_label or MODEL_NAMES.get(model_name, model_name)

    results = run_loso(df, feature_cols, model_name=model_name,
                        only_high_confidence=only_high_confidence,
                        random_state=random_state, return_results=True)

    if results is None:
        print(f"  (sin resultados: no se generaron graficas en {output_dir})")
        return None

    save_confusion_matrix(results["cm"], results["labels"], output_dir / "confusion_matrix.png", title=f"Matriz de confusion - {label}")
    save_feature_importance(results["imp_df"], output_dir / "feature_importance.png", top_n=top_n_features, title=f"Importancia de features - {label}")
    save_class_metrics(results["report_dict"], output_dir / "class_metrics.png", title=f"Precision / Recall / F1 por clase - {label}")
    save_fold_accuracy(results["fold_results"], output_dir / "fold_accuracy.png", title=f"Accuracy por fold - {label}")
    save_metrics_json(results, output_dir / "metrics.json", model_label=label, scenario=output_dir.name)

    print(f"  -> Graficas guardadas en: {output_dir}")
    return results


def generate_cnn_plots(results, output_dir, model_label="CNN"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results is None:
        print(f"  (sin resultados: no se generaron graficas en {output_dir})")
        return None

    save_confusion_matrix(results["cm"], results["labels"], output_dir / "confusion_matrix.png", title=f"Matriz de confusion - {model_label}")
    save_class_metrics(results["report_dict"], output_dir / "class_metrics.png", title=f"Precision / Recall / F1 por clase - {model_label}")
    save_fold_accuracy(results["fold_results"], output_dir / "fold_accuracy.png", title=f"Accuracy por fold - {model_label}")
    save_metrics_json(results, output_dir / "metrics.json", model_label=model_label, scenario=output_dir.name)

    print(f"  -> Graficas guardadas en: {output_dir}")
    return results