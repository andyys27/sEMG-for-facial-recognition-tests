"""
Entrena y evalua un clasificador Random Forest sobre el dataset de
features (logica de load/train en emg_ml.model)

python -m main.train_baseline
"""

from pathlib import Path
from .emg_ml import load_dataset, run_loso
from .emg_ml.plotting_ml import plot_ml_summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"
    dataset_path.parent.mkdir(exist_ok=True)

    results_dir = root / "Results"
    results_dir.mkdir(exist_ok=True)

    df, feature_cols = load_dataset(dataset_path)

    print(f"Dataset: {len(df)} ventanas, {len(feature_cols)} features, "
          f"sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}")

    print("\n" + "#" * 60)
    print("# Evaluacion 1: entrenando con TODO (incluye baja confianza)")
    print("#" * 60)
    run_loso(df, feature_cols, only_high_confidence=False)

    imp_df1, y_true1, y_pred1, fold_accs1 = run_loso(df, feature_cols, only_high_confidence=False)
    if imp_df1 is not None:
        plot_ml_summary(
            y_true=y_true1,
            y_pred=y_pred1,
            imp_df=imp_df1,
            fold_results=fold_accs1,
            output_dir=results_dir,
            dpi=300,
            prefix="eval_completa"
        )

    print("\n" + "#" * 60)
    print("# Evaluacion 2: entrenando SOLO con alta confianza")
    print("#" * 60)
    run_loso(df, feature_cols, only_high_confidence=True)

    