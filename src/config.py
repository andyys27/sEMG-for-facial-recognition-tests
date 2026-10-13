"""Utilidades de configuración (YAML en config/)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_yaml(relative_path: str) -> dict:
    """Carga un YAML relativo a config/, p. ej. load_yaml('protocol/events.yaml')."""
    with open(CONFIG_DIR / relative_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}