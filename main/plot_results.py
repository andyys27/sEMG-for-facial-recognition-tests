"""
Genera y guarda las graficas de evaluacion del modelo Random Forest (LOSO)
sobre el dataset de features actual: matriz de confusion, importancia de
features, metricas por clase y accuracy por fold.

python -m main.plot_results
"""

from pathlib import Path
from .emg_ml import load_dataset
from .emg_ml.plots import generate_all_plots


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"
    output_dir = root / "Results" / "RandomForest_LOSO"

    df, feature_cols = load_dataset(dataset_path)

    print(f"Dataset: {len(df)} ventanas, {len(feature_cols)} features, "
          f"sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}")

    print("\n# Gráficas 1: Entrenando con TODO (incluye baja confianza)")
    generate_all_plots(df, feature_cols, output_dir / "all_data", only_high_confidence=False)

    print("\n# Gráficas 2: Entrenando SOLO con alta confianza")
    generate_all_plots(df, feature_cols, output_dir / "high_confidence", only_high_confidence=True)