"""Capa de entrada/salida: lee archivos y sincroniza. NO define etiquetas."""
from .load_emg import load_emg, load_imu
from .load_events import load_events, parse_detail
from .sync_timestamps import sync_timestamps, load_session, Session

__all__ = ["load_emg", "load_imu", "load_events", "parse_detail",
           "sync_timestamps", "load_session", "Session"]
