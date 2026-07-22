"""
Entrena y evalua varios clasificadores tabulares (Random Forest, SVM, KNN,
LDA, MLP) sobre el dataset de features, cada uno con LOSO completo

Ejemplos:
    python -m main.train_models                             
    python -m main.train_models --models rf svm             
    python -m main.train_models --models mlp --scenario high_confidence
    python -m main.train_models --scenario both            
"""

import argparse
from pathlib import Path
 
from .emg_ml import load_dataset
from .emg_ml.models import MODEL_NAMES
from .emg_ml.plots import generate_all_plots
 
 
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"], help=f"Modelos a correr: {list(MODEL_NAMES)} o 'all' (default: all)")
    parser.add_argument("--scenario", choices=["all_data", "high_confidence", "both"], default="all_data", help="Que escenario de confianza correr (default: all_data)")
    parser.add_argument("--list", action="store_true", help="Solo imprime los modelos disponibles y termina")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
 
    if args.list:
        print("Modelos disponibles:")
        for key, label in MODEL_NAMES.items():
            print(f"  {key:6s} -> {label}")
        raise SystemExit(0)
 
    if "all" in args.models:
        models_to_run = list(MODEL_NAMES.keys())
    else:
        invalid = [m for m in args.models if m not in MODEL_NAMES]
        if invalid:
            raise SystemExit(f"Modelo(s) desconocido(s): {invalid}. Validos: {list(MODEL_NAMES)}")
        models_to_run = args.models
 
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "Dataset" / "features_dataset.csv"
    results_root = root / "Results"
 
    df, feature_cols = load_dataset(dataset_path)
 
    print(f"Dataset: {len(df)} ventanas, {len(feature_cols)} features, "
          f"sujetos={sorted(df['Subject'].unique())}")
    print(f"Clases: {sorted(df['Label'].unique())}")
    print(f"Modelos a correr: {[MODEL_NAMES[m] for m in models_to_run]}")
    print(f"Escenario(s): {args.scenario}")
 
    for model_key in models_to_run:
        model_label = MODEL_NAMES[model_key]
        output_dir = results_root / f"{model_label}_LOSO"
 
        print(f"\n{'=' * 70}\nModelo: {model_label}\n{'=' * 70}")
 
        if args.scenario in ("all_data", "both"):
            print(f"\n# {model_label} - entrenando con TODO (incluye baja confianza)")
            generate_all_plots(df, feature_cols, output_dir / "all_data", model_name=model_key, model_label=model_label, only_high_confidence=False)
 
        if args.scenario in ("high_confidence", "both"):
            print(f"\n# {model_label} - entrenando SOLO con alta confianza")
            generate_all_plots(df, feature_cols, output_dir / "high_confidence", model_name=model_key, model_label=model_label, only_high_confidence=True)