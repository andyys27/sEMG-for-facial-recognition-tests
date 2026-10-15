"""Serialización de modelos."""
from pathlib import Path
import joblib


def save_model(model, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta or {}}, path)
    return path


def load_model(path: str | Path):
    """Devuelve (modelo, meta)."""
    d = joblib.load(path)
    return d["model"], d["meta"]