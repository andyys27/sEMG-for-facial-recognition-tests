"""
Construye el dataset de features (una fila por ventana) a partir de los
processed_labeled.csv de cada sujeto ya generados por run_segmentation.py

python -m main.build_dataset
"""

from pathlib import Path
from .emg_ml import DatasetConfig, build_dataset


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    output_path = root / "Dataset" / "features_dataset.csv"
    output_path.parent.mkdir(exist_ok=True)

    cfg = DatasetConfig(
        channel_groups={
            "Grupo_A": ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS"],
            "Grupo_B": ["EMG3_Envelope_RMS", "EMG4_Envelope_RMS"],
        },
        window_sec=0.25,
        overlap=0.5,
        exclude_countdown=True,
        include_reposo=True,
    )

    # Agrega aqui cada sujeto que tengas procesado
    subject_csv_map = {
        "sujeto1": root / "Test1" / "Data" / "processed_labeled.csv",
        "sujeto2": root / "Test2" / "Data" / "processed_labeled.csv",
    }

    dataset = build_dataset(subject_csv_map, cfg, output_path=output_path)