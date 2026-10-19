from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from desafio_logistica.data import build_capacity_wide, build_daily, build_forecast_table, load_data

ZIP = Path(__file__).resolve().parents[1] / "data" / "data_package_desafio_logistica.zip"

warnings.filterwarnings("ignore", message="Skipping features without any observed values")


@pytest.fixture(scope="session")
def tables():
    """Los 4 CSV del data package más las tablas derivadas (demanda diaria, capacidad y pronóstico con real)."""
    if not ZIP.exists():
        pytest.skip("No está el data package en data/")
    d = load_data(ZIP)
    daily = build_daily(d["pedidos"])
    return {**d, "daily": daily, "cap_w": build_capacity_wide(d["cap"]), "fc": build_forecast_table(d["pron"], daily)}
