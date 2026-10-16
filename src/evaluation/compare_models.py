"""Comparación entre modelos: tabla, gráfica y significancia pareada (Wilcoxon + Holm)."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import wilcoxon

from ..config import load_yaml
from ..models.common import models_root


def load_runs(root: str | Path | None = None) -> dict[str, dict]:
    root = Path(root) if root else models_root()
    runs = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/metrics.json"))}
    if not runs:
        raise FileNotFoundError(f"No hay resultados en {root}; ejecuta scripts.train_models primero")
    return runs


def comparison_table(runs: dict) -> pd.DataFrame:
    rows = []
    for name, r in runs.items():
        p, fs, sg = r["pooled"], r["fold_stats"], r["segment_vote"]
        rows.append({"model": name, "macro_f1": p["macro_f1"],
                     "macro_f1_fold_mean": fs["macro_f1"]["mean"], "macro_f1_fold_std": fs["macro_f1"]["std"],
                     "balanced_acc": p["balanced_accuracy"], "accuracy": p["accuracy"],
                     "weighted_f1": p["weighted_f1"], "seg_macro_f1": sg["macro_f1"],
                     "seg_accuracy": sg["accuracy"], "n_folds": r["n_folds"]})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def per_class_f1(runs: dict) -> pd.DataFrame:
    return pd.DataFrame({n: {c: v["f1"] for c, v in r["pooled"]["per_class"].items()}
                         for n, r in runs.items()}).T


def _holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def pairwise_wilcoxon(runs: dict) -> pd.DataFrame:
    """Wilcoxon pareado sobre el macro-F1 de cada fold, con corrección de Holm.
    `min_possible_p` = 2 / 2^n_folds: si es > 0.05, ninguna comparación puede ser significativa."""
    folds = {n: [r["per_fold"][f]["macro_f1"] for f in r["meta"]["folds"]] for n, r in runs.items()}
    rows = []
    for a, b in combinations(sorted(folds), 2):
        x, y = np.array(folds[a]), np.array(folds[b])
        try:
            p = 1.0 if np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
        except ValueError:
            p = float("nan")
        rows.append({"a": a, "b": b, "mean_diff_f1": float((x - y).mean()), "p": p,
                     "min_possible_p": 2 / 2 ** len(x)})
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out.loc[ok, "p_holm"] = _holm(out.loc[ok, "p"])
    return out


def plot_comparison(table: pd.DataFrame, n_classes: int, out_path: str | Path) -> None:
    fig = Figure(figsize=(9, 0.95 * len(table) + 1.8), layout="constrained")
    ax = fig.subplots()
    y, h = np.arange(len(table)), 0.38
    ax.barh(y - h / 2, table["macro_f1"], h, xerr=table["macro_f1_fold_std"], capsize=3, color="#6baed6",
            label="por ventana (± desv. entre folds)")
    ax.barh(y + h / 2, table["seg_macro_f1"], h, color="#fd8d3c", label="por segmento (voto mayoritario)")
    ax.axvline(1 / n_classes, ls="--", color="gray", lw=1, label=f"azar ({1 / n_classes:.2f})")
    ax.set_yticks(y, table["model"]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1"); ax.set_title("Comparación de modelos"); ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_path, dpi=110)


def write_comparison(models_dir: str | Path | None = None, out_dir: str | Path | None = None) -> dict:
    """Escribe model_comparison.{csv,png,md} en data/analysis/reports/."""
    runs = load_runs(models_dir)
    out = Path(out_dir) if out_dir else Path(load_yaml("experiment/settings.yaml")["paths"]["analysis"]) / "reports"
    out.mkdir(parents=True, exist_ok=True)

    table, pc, sig = comparison_table(runs), per_class_f1(runs), pairwise_wilcoxon(runs)
    table.to_csv(out / "model_comparison.csv", index=False)
    n_classes = len(next(iter(runs.values()))["meta"]["classes"])
    plot_comparison(table, n_classes, out / "model_comparison.png")

    n = int(table["n_folds"].iloc[0])
    meta = next(iter(runs.values()))["meta"]
    lines = ["# Comparación de modelos\n",
             f"Validación: **{n} folds** ({', '.join(meta['folds'])}) | balanceo de entrenamiento: "
             f"**{meta['balance_train']}** | {meta['n_windows']} ventanas, {meta['n_features']} features\n",
             "## Métricas (ordenado por macro-F1 agregado)\n", "```", table.round(3).to_string(index=False), "```\n",
             "## F1 por clase (agregado de todos los folds)\n", "```", pc.round(2).to_string(), "```\n",
             "## Significancia (Wilcoxon pareado sobre macro-F1 por fold, corrección de Holm)\n"]
    if len(sig):
        lines += ["```", sig.round(4).to_string(index=False), "```\n"]
        if sig["min_possible_p"].iloc[0] > 0.05:
            lines.append(f"> Con {n} folds el p-valor más pequeño posible es {sig['min_possible_p'].iloc[0]:.4f} "
                         f"(> 0.05): **ninguna diferencia puede salir significativa**. Los p-valores son "
                         f"orientativos hasta tener más folds/sujetos.\n")
    p_md = out / "model_comparison.md"
    p_md.write_text("\n".join(lines), encoding="utf-8")
    return {"table": table, "per_class": pc, "significance": sig, "dir": out, "summary": p_md}