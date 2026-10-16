"""Métricas y gráficas de evaluación.
Nota: compare_models y diagnose_generalization se importan por su ruta (dependen de src.models)."""
from .metrics import classification_metrics, summarize, segment_vote, class_info
from .confusion_matrix import plot_confusion_matrix

__all__ = ["classification_metrics", "summarize", "segment_vote", "class_info", "plot_confusion_matrix"]