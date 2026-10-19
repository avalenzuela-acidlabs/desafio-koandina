"""Simulación de aplanamiento de la curva de demanda (Tarea 2, punto 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def flatten_simulation(
    daily_units: pd.DataFrame,
    capacity: pd.DataFrame,
    alpha: float,
    trigger: float = 0.90,
    window: int = 2,
) -> pd.DataFrame:
    """Peak-shaving greedy sobre la demanda histórica.

    Mueve hasta ``alpha`` × unidades de cada día con utilización > ``trigger`` hacia días vecinos
    (±``window``) con holgura bajo ``trigger``. Devuelve la demanda redistribuida (fecha × CD).
    """
    out = daily_units.copy()
    for cd in out.columns:
        u = out[cd].to_numpy(dtype=float).copy()
        c = capacity[cd].reindex(out.index).to_numpy(dtype=float)
        for i in np.argsort(-(u / c)):
            excess = u[i] - trigger * c[i]
            if excess <= 0:
                continue
            move = min(alpha * u[i], excess)
            neighbors = [j for j in range(max(0, i - window), min(len(u), i + window + 1)) if j != i]
            neighbors.sort(key=lambda j: u[j] / c[j])
            for j in neighbors:
                room = trigger * c[j] - u[j]
                if room <= 0 or move <= 0:
                    continue
                q = min(room, move)
                u[i] -= q
                u[j] += q
                move -= q
        out[cd] = u
    return out


def load_profile(d: pd.DataFrame, c: pd.DataFrame) -> dict[str, float]:
    """KPIs de aplanamiento: utilización alta, peak-to-average semanal, dispersión y peso del martes."""
    util = d / c
    weekly = d.groupby(d.index.to_period("W"))
    par = weekly.transform("max") / weekly.transform("mean")
    return {
        "%CD-días util>90%": float((util > 0.9).to_numpy().mean()),
        "%CD-días util>100%": float((util > 1.0).to_numpy().mean()),
        "PAR semanal medio (máx/prom)": float(par.to_numpy().mean()),
        "CV de utilización": float(util.stack().std() / util.stack().mean()),
        "martes: % del volumen semanal": float(d[d.index.dayofweek == 1].sum().sum() / d.sum().sum()),
    }


def simulate_alphas(
    daily: pd.DataFrame, capacity: pd.DataFrame, alphas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
) -> pd.DataFrame:
    """Perfil de carga para distintas fracciones ``alpha`` de carga excedente que acepta moverse."""
    return pd.DataFrame(
        {f"α={a:.0%}": load_profile(flatten_simulation(daily, capacity, a), capacity) for a in alphas}
    ).T
