from __future__ import annotations

import pandas as pd


def train_model(dataset: pd.DataFrame, target_col: str = "label") -> dict:
    """Esqueleto de entrenamiento para el dataset final.

    En la implementación real aquí se instanciaría el modelo y se calcularían
    métricas LOSO o de validación cruzada según el caso.
    """
    if dataset.empty:
        raise ValueError("El dataset está vacío.")

    return {
        "rows": len(dataset),
        "target": target_col,
        "status": "pipeline_ready",
    }
