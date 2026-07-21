"""
Entrena y evalua un clasificador Random Forest sobre el dataset de
features (logica de load/train en emg_ml.model)

python -m main.train_baseline
"""

from pathlib import Path
from .emg_ml import load_dataset, run_loso


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"
    dataset_path.parent.mkdir(exist_ok=True)

    df, feature_cols = load_dataset(dataset_path)

    print(f"Dataset: {len(df)} ventanas, {len(feature_cols)} features, "
          f"sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}")

    print("# Evaluacion 1: entrenando con TODO (incluye baja confianza)")
    run_loso(df, feature_cols, only_high_confidence=False)

    print("# Evaluacion 2: entrenando SOLO con alta confianza")
    run_loso(df, feature_cols, only_high_confidence=True)