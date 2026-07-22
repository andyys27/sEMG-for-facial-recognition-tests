"""
Entrena y evalua varios clasificadores tabulares (Random Forest, SVM, KNN,
LDA, MLP) sobre el dataset de features, cada uno con LOSO completo (todo el
dataset + solo alta confianza), guardando sus graficas en
Results/<Modelo>_LOSO/{todo,alta_confianza}/

python -m main.train_models
"""

from pathlib import Path
from .emg_ml import load_dataset
from .emg_ml.models import MODEL_NAMES
from .emg_ml.plots import generate_all_plots


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"
    results_root = root / "Results"

    df, feature_cols = load_dataset(dataset_path)

    print(f"Dataset: {len(df)} ventanas, {len(feature_cols)} features, "
          f"sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}")

    for model_key, model_label in MODEL_NAMES.items():
        output_dir = results_root / f"{model_label}_LOSO"

        print(f"\n{'=' * 70}\nModelo: {model_label}\n{'=' * 70}")

        print(f"\n# {model_label} - entrenando con TODO (incluye baja confianza)")
        generate_all_plots(df, feature_cols, output_dir / "all_data", model_name=model_key, model_label=model_label, only_high_confidence=False)

        print(f"\n# {model_label} - entrenando SOLO con alta confianza")
        generate_all_plots(df, feature_cols, output_dir / "high_confidence", model_name=model_key, model_label=model_label, only_high_confidence=True)