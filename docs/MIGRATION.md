# Mapa de migracion: codigo actual -> nueva estructura

| Codigo actual | Destino | Fase |
|---|---|---|
| `main/processing.py` (filtros, envolvente) | `src/segmentation/` o `src/features/` (preproceso) | 4-5 |
| `main/pipeline.py`, `main/main.py` | `src/pipeline/` + `scripts/prepare_data.py` | 7 |
| `main/emg_segmentation/labeling.py` | **Reemplazado** por `src/protocol/` (labels desde Unity) | 3 |
| `main/emg_segmentation/detection.py`, `epochs.py` | `src/segmentation/validate_segment.py` (solo validacion) | 4 |
| `main/emg_ml/features.py` | `src/features/*` | 5 |
| `main/emg_ml/dataset.py`, `raw_dataset.py`, `build_dataset.py` | `src/dataset/` | 5 |
| `main/emg_ml/model.py`, `models.py`, `train_models.py` | `src/models/` | 6 |
| `main/emg_ml/cnn_model.py`, `train_cnn.py` | `src/models/cnn_model.py` | 6 |
| `main/compare_*.py`, `diagnose_generalization.py`, `error_analysis.py` | `src/evaluation/` | 6 |
| `Test1/`, `Test2/` | `data/raw/subject_01/session_01/`, `subject_02/...` | 2 |
