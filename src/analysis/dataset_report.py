"""Reportes entre sesiones/sujetos -> data/analysis/reports/."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..config import load_yaml
from ..dataset.build_dataset import feature_columns
from .signal_diagnostics import ORDER, COLORS


def _plot_class_balance(df: pd.DataFrame, path: Path) -> None:
    tab = pd.crosstab(df["subject"], df["label"]).reindex(columns=[l for l in ORDER if l in set(df.label)])
    fig = Figure(figsize=(8, 4), layout="constrained")
    ax = fig.subplots()
    w = 0.8 / len(tab.index)
    for i, (subj, row) in enumerate(tab.iterrows()):
        ax.bar(np.arange(len(row)) + i * w, row.values, w, label=subj, color=None)
    ax.set_xticks(np.arange(tab.shape[1]) + 0.4 - w / 2, tab.columns)
    ax.set_ylabel("ventanas"); ax.set_title("Balance de clases"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=110)


def _plot_feature_class_heatmap(df: pd.DataFrame, path: Path) -> None:
    cols = [c for c in feature_columns(df) if c.endswith(("_RMS", "_MNF", "_ZC"))
            or c.startswith(("ratio_", "diff_"))]
    z = (df[cols] - df[cols].mean()) / (df[cols].std() + 1e-30)
    tab = z.groupby(df["label"]).mean().reindex([l for l in ORDER if l in set(df.label)])
    fig = Figure(figsize=(max(8, 0.55 * len(cols)), 4.2), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(tab.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8); ax.set_yticks(range(len(tab)), tab.index)
    fig.colorbar(im, ax=ax, label="media z-score por clase")
    ax.set_title("Qué features separan cada clase")
    fig.savefig(path, dpi=110)


def write_dataset_report(df: pd.DataFrame, folds, out_dir: str | Path | None = None) -> dict:
    """Escribe dataset_summary.md, class_balance.png y feature_class_heatmap.png."""
    root = out_dir or Path(load_yaml("experiment/settings.yaml")["paths"]["analysis"]) / "reports"
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)

    tab = pd.crosstab(df["subject"], df["label"])
    n_feat = len(feature_columns(df))
    lines = [f"# Resumen del dataset\n",
             f"- Ventanas: **{len(df)}**  |  Features: **{n_feat}**  |  Sujetos: **{df.subject.nunique()}**"
             f"  |  Sesiones: **{df.groupby(['subject', 'session']).ngroups}**\n",
             "## Ventanas por sujeto y etiqueta\n", "```", tab.to_string(), "```\n",
             "## Folds de evaluación\n", "```"]
    for f in folds:
        cls = df.iloc[f.test]["label"].value_counts().to_dict()
        lines.append(f"{f.name:<14} train={len(f.train):>5}  test={len(f.test):>5}  test por clase={cls}")
    lines += ["```\n"]
    p_md = out / "dataset_summary.md"
    p_md.write_text("\n".join(lines), encoding="utf-8")

    _plot_class_balance(df, out / "class_balance.png")
    _plot_feature_class_heatmap(df, out / "feature_class_heatmap.png")
    return {"dir": out, "summary": p_md}