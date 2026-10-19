"""Optimización logística: diagnóstico, corrección de pronóstico y Gate de historia mínima."""
from __future__ import annotations

from .config import MIN_HISTORY_DAYS
from .gate import GateDecision, evaluate_history_gate, gated_forecast
from .metrics import error_metrics

__all__ = [
    "MIN_HISTORY_DAYS",
    "GateDecision",
    "error_metrics",
    "evaluate_history_gate",
    "gated_forecast",
]
__version__ = "0.1.0"
