"""
Graficas de evaluacion de aprendizaje automatico (sEMG):
  - plot_confusion_matrix: matriz de confusion en mapa de calor con frecuencias y porcentajes
  - plot_feature_importances: top N caracteristicas mas relevantes en formato horizontal
  - plot_loso_accuracy: desempeno del modelo por sujeto/fold + linea de promedio global
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Paleta de colores consistente con el resto del proyecto
EMOTION_COLORS = {
    "Reposo": "gray",
    "Disgusto": "red",    
    "Sonrisa": "green",    
    "Sorprendido": "blue", 
    "Triste": "purple",      
}


def plot_confusion_matrix(y_true, y_pred, classes=None, output_dir="Results", dpi=300, prefix="loso"):    
    # Genera y guarda un mapa de calor de la matriz de confusion con conteo y porcentaje por fila
    outhpath = Path(output_dir)
    outhpath.mkdir(parents=True, exist_ok=True)

    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Barra de color
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Número de Ventanas", fontsize=9, weight="bold")

    # Configuracion de ejes
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9.5, weight="bold")
    ax.set_yticklabels(classes, fontsize=9.5, weight="bold")

    # Formato de celdas (Texto con cantidad + porcentaje)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            perc = cm_perc[i, j] * 100
            text_color = "white" if val > thresh else "black"
            ax.text(
                j, i, f"{val}\n({perc:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=8.5, weight="bold"
            )

    ax.set_title(f"Matriz de Confusión Matriz LOSO — {prefix.capitalize()}", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Predicción del Modelo", fontsize=10, weight="bold")
    ax.set_ylabel("Gesto Real (Ground Truth)", fontsize=10, weight="bold")
    ax.grid(False)
    fig.tight_layout()

    out = outhpath / f"confusion_matrix_{prefix}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_feature_importances(imp_df, top_n=15, output_dir="Results", dpi=300, prefix="loso"):
    # Genera y guarda un grafico de barras horizontales con el Top N de características mas importantes
    outhpath = Path(output_dir)
    outhpath.mkdir(parents=True, exist_ok=True)

    # Preparar datos
    if isinstance(imp_df, pd.Series):
        top_df = imp_df.head(top_n).reset_index()
        top_df.columns = ["Feature", "Importance"]
    else:
        top_df = imp_df.head(top_n).copy()
        if "Feature" not in top_df.columns:
            top_df = top_df.reset_index()
            top_df.columns = ["Feature", "Importance"]

    top_df = top_df.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Renderizado de barras
    bars = ax.barh(top_df["Feature"], top_df["Importance"], color="gray", height=0.65, edgecolor="none")

    # Resaltar las top 3 con un color distintivo
    for i, bar in enumerate(bars[-3:]):
        bar.set_color("orange")

    # Etiquetas de valor en cada barra
    max_val = top_df["Importance"].max()
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max_val * 0.01), bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center", ha="left", fontsize=8, weight="bold", color="gray"
        )

    ax.set_title(f"Top {top_n} Características más Relevantes — {prefix.capitalize()}", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Importancia Promedio (Gini / Módulo RF)", fontsize=9.5, weight="bold")
    ax.set_ylabel("Característica sEMG", fontsize=9.5, weight="bold")
    ax.set_xlim(0, max_val * 1.15)
    ax.grid(True, axis="x", alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = outhpath / f"top_features_{prefix}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_loso_accuracy(fold_results, output_dir="Results", dpi=300, prefix="loso"):
    # Genera y guarda un grafico de barras con el Accuracy obtenido en cada Fold/Sujeto LOSO
    outhpath = Path(output_dir)
    outhpath.mkdir(parents=True, exist_ok=True)

    if isinstance(fold_results, dict):
        subjects = list(fold_results.keys())
        accs = list(fold_results.values())
    else:
        subjects = [f[0] for f in fold_results]
        accs = [f[1] for f in fold_results]

    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    x = np.arange(len(subjects))
    bars = ax.bar(x, [a * 100 for a in accs], color="gray", width=0.45, alpha=0.9, edgecolor="black", lw=0.8)

    # Linea de Promedio Global
    ax.axhline(
        mean_acc * 100, color="red", ls="--", lw=1.5,
        label=f"Promedio Global = {mean_acc * 100:.1f}%"
    )

    # Anotar valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 1.5,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=9, weight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold: {s}" for s in subjects], fontsize=9.5, weight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_title(f"Exactitud por Sujeto en Validación LOSO — {prefix.capitalize()}", fontsize=11.5, weight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = outhpath / f"accuracy_loso_{prefix}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_ml_summary(y_true, y_pred, imp_df, fold_results, output_dir="Results", dpi=300, prefix="loso"):
    # Llama a las tres funciones de visualizacion para exportar todo el paquete de graficas
    print(f"\nGenerando gráficas de diagnóstico ML para '{prefix}'...")
    plot_confusion_matrix(y_true, y_pred, output_dir=output_dir, dpi=dpi, prefix=prefix)
    plot_feature_importances(imp_df, top_n=15, output_dir=output_dir, dpi=dpi, prefix=prefix)
    plot_loso_accuracy(fold_results, output_dir=output_dir, dpi=dpi, prefix=prefix)