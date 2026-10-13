"""Lectura de CSV crudo de EMG (FreeEMG) e IMU (Baiobit)."""
from pathlib import Path
import pandas as pd
from ._time import timestamp_to_ns

EMG_CHANNELS = ["EMG1", "EMG2", "EMG3", "EMG4"]
IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"]


def _read(path, value_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ["Timestamp", *value_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}; encontradas {list(df.columns)}")
    df.insert(0, "t_ns", timestamp_to_ns(df["Timestamp"]))
    return df[["t_ns", "Timestamp", *value_cols]]


def load_emg(path: str | Path, channels: list[str] = EMG_CHANNELS) -> pd.DataFrame:
    """FREEEMG_EMG_with_timestamp.csv -> ['t_ns', 'Timestamp', 'EMG1'..'EMG4']."""
    return _read(path, channels)


def load_imu(path: str | Path) -> pd.DataFrame:
    """Baiobit_with_timestamp.csv -> ['t_ns', 'Timestamp', ax..yaw]."""
    return _read(path, IMU_COLUMNS)
