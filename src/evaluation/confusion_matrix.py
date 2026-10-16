"""Matriz de confusión (normalizada por fila, con conteos)."""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure


def plot_confusion_matrix(cm, names: list[str], title: str, out_path: str | Path) -> None:
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig = Figure(figsize=(6.2, 5.2), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xlabel("predicho"); ax.set_ylabel("real"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="recall (fila)")
    fig.savefig(out_path, dpi=110)