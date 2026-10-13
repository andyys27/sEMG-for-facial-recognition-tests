"""Diagnóstico de datos: gráficas y reportes que se guardan en data/analysis."""
from .session_report import analyze_session, quality_report
from .dataset_report import write_dataset_report

__all__ = ["analyze_session", "quality_report", "write_dataset_report"]