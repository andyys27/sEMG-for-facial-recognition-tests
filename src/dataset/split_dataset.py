"""Splits sin fuga: por sujeto (LOSO) o, con un solo sujeto, por repetición."""
from typing import Iterator, NamedTuple
import numpy as np
import pandas as pd

from ..config import load_yaml


class Fold(NamedTuple):
    name: str
    train: np.ndarray        # posiciones (iloc) de las filas de entrenamiento
    test: np.ndarray


def get_splits(df: pd.DataFrame, strategy: str | None = None) -> Iterator[Fold]:
    """strategy: 'loso' | 'leave_one_rep_out' | 'auto' (default de config/dataset/split.yaml).

    - loso: cada fold prueba con un sujeto completo (requiere >= 2 sujetos).
    - leave_one_rep_out: cada fold prueba con una repetición del protocolo. Las ventanas
      de un mismo segmento nunca se separan. El reposo inicial (rep = -1) siempre va
      a entrenamiento porque no tiene gesto asociado.
    """
    strategy = strategy or load_yaml("dataset/split.yaml").get("strategy", "auto")
    n_subj = df["subject"].nunique()
    if strategy == "auto":
        strategy = "loso" if n_subj >= 2 else "leave_one_rep_out"

    if strategy == "loso":
        if n_subj < 2:
            raise ValueError("LOSO requiere al menos 2 sujetos")
        for s in sorted(df["subject"].unique()):
            test = (df["subject"] == s).to_numpy()
            yield Fold(f"subject={s}", np.flatnonzero(~test), np.flatnonzero(test))
    elif strategy == "leave_one_rep_out":
        for r in sorted(r for r in df["rep"].unique() if r >= 0):
            test = (df["rep"] == r).to_numpy()
            yield Fold(f"rep={r}", np.flatnonzero(~test), np.flatnonzero(test))
    else:
        raise ValueError(f"strategy desconocida: {strategy}")