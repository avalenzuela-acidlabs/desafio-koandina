"""Constantes compartidas por todo el paquete."""
from __future__ import annotations

SEED = 42
MIN_HISTORY_DAYS = 45  # umbral del Gate de historia mínima (Tarea 4)
ZIP_NAME = "data_package_desafio_logistica.zip"

DOW_ES: list[str] = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
HORIZONS: list[str] = ["D+1", "D+2", "D+3", "D+4"]

# Modelo de corrección (Tarea 3)
RATIO_BOUNDS: tuple[float, float] = (0.5, 2.0)  # guardrail sobre fc_corregido / fc_base
PEAK_MULT = 1.15  # un día es "pico" si supera 15 % su media móvil de 14 días
RIDGE_ALPHA = 3.0

# Impacto en el negocio (supuestos explícitos, ajustables)
COST_PLAN = 1.00  # costo unitario de la flota contratada (se use o no)
COST_EMERGENCY = 1.80  # costo unitario del flete de emergencia
