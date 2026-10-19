"""Impacto en el negocio: flota, flete de emergencia y nivel de servicio."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST_EMERGENCY, COST_PLAN


def fleet_impact(
    df: pd.DataFrame,
    pred: str,
    capacity: pd.DataFrame,
    buffer: float = 0.0,
    cost_plan: float = COST_PLAN,
    cost_emergency: float = COST_EMERGENCY,
) -> pd.Series:
    """Costo y servicio de contratar flota según un pronóstico.

    ``contratado = min(pronóstico × (1 + colchón), capacidad)``. Lo que excede lo contratado se cubre con flete de
    emergencia y lo contratado no usado es capacidad ociosa. ``capacity`` es (fecha × CD).
    """
    idx = pd.MultiIndex.from_arrays([df.fecha_despacho, df.centro_distribucion])
    cap_row = pd.Series(capacity.stack().reindex(idx).to_numpy(), index=df.index)
    booked = np.minimum(df[pred] * (1 + buffer), cap_row)
    emerg = np.maximum(df.real - booked, 0.0)
    idle = np.maximum(booked - df.real, 0.0)
    return pd.Series(
        {
            "unid_emergencia": emerg.sum(),
            "unid_ociosas": idle.sum(),
            "%_flete_emergencia": emerg.sum() / df.real.sum(),
            "%_capacidad_ociosa": idle.sum() / booked.sum(),
            "fill_rate": 1 - emerg.sum() / df.real.sum(),
            "costo_total_idx": cost_plan * booked.sum() + cost_emergency * emerg.sum(),
        }
    )


def impact_table(
    bt: pd.DataFrame, capacity: pd.DataFrame, buffers: tuple[float, ...] = (0.0, 0.05, 0.10)
) -> pd.DataFrame:
    """Impacto del pronóstico base vs el corregido (Ridge) para distintos colchones de flota."""
    rows = {}
    for buf in buffers:
        for col, label in [("pred_base", "Base"), ("pred_ridge", "Corregido")]:
            rows[(f"colchón {buf:.0%}", label)] = fleet_impact(bt, col, capacity, buf)
    return pd.DataFrame(rows).T


def relative_reduction(impact: pd.DataFrame) -> pd.DataFrame:
    """Variación relativa Base → Corregido por escenario de colchón."""
    rows = []
    for scenario in impact.index.get_level_values(0).unique():
        b, c = impact.loc[(scenario, "Base")], impact.loc[(scenario, "Corregido")]
        rows.append(
            {
                "escenario": scenario,
                "Δ unid. emergencia": c.unid_emergencia / b.unid_emergencia - 1,
                "Δ unid. ociosas": c.unid_ociosas / b.unid_ociosas - 1,
                "Δ costo total": c.costo_total_idx / b.costo_total_idx - 1,
                "Δ fill rate (pp)": (c.fill_rate - b.fill_rate) * 100,
            }
        )
    return pd.DataFrame(rows).set_index("escenario")
