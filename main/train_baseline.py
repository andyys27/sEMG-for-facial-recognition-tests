"""
Entrena y evalua un clasificador Random Forest sobre el dataset de
features (logica de load/train en emg_ml.model)

python -m main.train_baseline.py
"""

from emg_ml import load_dataset, run_loso


if __name__ == "__main__":
    dataset_path = "../Dataset/features_dataset.csv"
    df, feature_cols = load_dataset(dataset_path)

    print(f"Dataset: {len(df)} ventanas, {len(feature_cols)} features, "
          f"sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}")

    print("\n" + "#" * 60)
    print("# Evaluacion 1: entrenando con TODO (incluye baja confianza)")
    print("#" * 60)
    run_loso(df, feature_cols, only_high_confidence=False)

    print("\n\n" + "#" * 60)
    print("# Evaluacion 2: entrenando SOLO con alta confianza")
    print("#" * 60)
    run_loso(df, feature_cols, only_high_confidence=True)