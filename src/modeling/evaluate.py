from __future__ import annotations

import pandas as pd


def evaluate_model(model_output: dict, dataset: pd.DataFrame) -> dict:
    """Esqueleto para evaluación del modelo."""
    return {
        "rows": len(dataset),
        "model_status": model_output.get("status", "unknown"),
        "evaluation": "pending_implementation",
    }
