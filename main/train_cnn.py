"""
Construye el dataset de ventanas crudas (EMG*_Filtered) y entrena/evalua 
el CNN 1D con LOSO completo

python -m main.train_cnn
"""

import numpy as np
from pathlib import Path

from .emg_ml.raw_dataset import build_raw_dataset, RawDatasetConfig
from .emg_ml.cnn_model import run_loso_cnn
from .emg_ml.plots import generate_cnn_plots


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    raw_dataset_path = root / "Dataset" / "raw_windows.npz"
    results_root = root / "Results"

    cfg = RawDatasetConfig(
        signal_type="Filtered",
        window_sec=0.25,
        overlap=0.5,
        exclude_countdown=True,
        include_reposo=True,
    )

    subject_specs = {
        "Test1": (root / "Test1" / "Data" / "processed_labeled.csv",
                  ["EMG1", "EMG4", "EMG2", "EMG3"]),   # Grupo_A=[1,4] + Grupo_B=[2,3]
        "Test2": (root / "Test2" / "Data" / "processed_labeled.csv",
                  ["EMG1", "EMG2", "EMG3", "EMG4"]),   # Grupo_A=[1,2] + Grupo_B=[3,4]
    }

    if raw_dataset_path.exists():
        npz = np.load(raw_dataset_path, allow_pickle=True)
        raw_data = {k: npz[k] for k in npz.files}
        print(f"Dataset crudo cargado desde cache: {raw_dataset_path}")
    else:
        raw_data = build_raw_dataset(subject_specs, cfg=cfg, output_path=raw_dataset_path)

    print(f"\nX shape: {raw_data['X'].shape}, clases: {sorted(set(raw_data['y']))}")

    for split_label, only_high in [("todo", False), ("alta_confianza", True)]:
        output_dir = results_root / "CNN_LOSO" / split_label
        print(f"\n{'=' * 70}\nCNN - {split_label}\n{'=' * 70}")
        results = run_loso_cnn(raw_data, only_high_confidence=only_high, return_results=True)
        if results is not None:
            generate_cnn_plots(results, output_dir)