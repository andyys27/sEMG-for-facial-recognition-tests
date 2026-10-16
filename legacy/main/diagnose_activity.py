"""
Diagnostico: revisa si las ventanas etiquetadas con una emocion activa
en realidad tienen actividad muscular similar a Reposo (linea base)

Si una fraccion grande de las ventanas de una clase "activa" tiene la 
misma amplitud que Reposo, el modelo NO esta fallando por falta de datos: 
esta prediciendo "Reposo" porque, en terminos de la senal, esas ventanas 
SI se parecen a Reposo

python -m main.diagnose_activity
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .emg_ml.model import load_dataset


def activity_score(df):
    env_cols = [c for c in df.columns if c.endswith("_env_mean")]
    if not env_cols:
        raise ValueError("No se encontraron columnas *_env_mean en el dataset. "
                          "Revisa el nombre real de tus columnas de envolvente en features.py")
    return df[env_cols].mean(axis=1)


def overlap_with_reposo(df, score_col="ActivityScore", reference_label="Reposo"):
    ref = df.loc[df["Label"] == reference_label, score_col]
    ref_p75 = ref.quantile(0.75)

    rows = []
    for label in sorted(df["Label"].unique()):
        if label == reference_label:
            continue
        vals = df.loc[df["Label"] == label, score_col]
        pct_below = float((vals <= ref_p75).mean() * 100)
        pooled_std = np.sqrt((vals.std() ** 2 + ref.std() ** 2) / 2)
        cohens_d = (vals.mean() - ref.mean()) / pooled_std if pooled_std > 0 else np.nan
        rows.append({
            "Clase": label,
            "N ventanas": len(vals),
            "Actividad media": vals.mean(),
            "Actividad Reposo (referencia)": ref.mean(),
            "% ventanas <= p75 de Reposo": pct_below,
            "Cohen's d vs Reposo": cohens_d,
        })
    return pd.DataFrame(rows).sort_values("% ventanas <= p75 de Reposo", ascending=False)


def plot_activity_histograms(df, score_col, save_path):
    labels = sorted(df["Label"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        vals = df.loc[df["Label"] == label, score_col]
        ax.hist(vals, bins=40, alpha=0.45, label=label, density=True)
    ax.set_xlabel("Actividad muscular (promedio env_mean, 4 canales)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion de actividad muscular por clase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"
    output_dir = root / "Results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, _ = load_dataset(dataset_path)
    df = df.copy()
    df["ActivityScore"] = activity_score(df)

    print("Actividad muscular (env_mean promedio) por clase:")
    print(df.groupby("Label")["ActivityScore"].describe().to_string())

    print("\nComparacion contra Reposo:")
    summary = overlap_with_reposo(df, "ActivityScore")
    print(summary.to_string(index=False))
    summary.to_csv(output_dir / "activity_vs_reposo.csv", index=False)

    plot_activity_histograms(df, "ActivityScore", output_dir / "activity_histograms.png")
    print(f"\n-> Histograma guardado: {output_dir / 'activity_histograms.png'}")

    print("\nInterpretacion:")
    for _, r in summary.iterrows():
        if r["% ventanas <= p75 de Reposo"] > 50:
            print(f"  {r['Clase']}: mas de la mitad de sus ventanas tienen actividad "
                  f"muscular igual o menor al 75% de las ventanas de Reposo -> muchas "
                  f"ventanas etiquetadas '{r['Clase']}' probablemente capturan "
                  f"transicion/gesto debil, no el gesto completo. Revisa la segmentacion "
                  f"(Phase/Block) o considera un umbral minimo de energia para "
                  f"descartar ventanas casi-en-reposo dentro de gestos activos.")
        elif abs(r["Cohen's d vs Reposo"]) < 0.3:
            print(f"  {r['Clase']}: actividad muy similar a Reposo (Cohen's d chico) -> "
                  f"la amplitud muscular sola no distingue esta clase de Reposo.")
        else:
            print(f"  {r['Clase']}: actividad claramente distinta a Reposo, esto no "
                  f"explica la confusion por si solo (revisa otras features).")