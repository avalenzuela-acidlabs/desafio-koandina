"""Ingeniería de features sin fuga temporal (Tarea 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PEAK_MULT

FEATURES_NUM: list[str] = [
    "lfc", "h", "es_martes", "es_finde", "spike_lag7", "fc_util",
    "rm7", "rm14", "rstd7", "fc_vs_rm7", "lag7_vs_rm14", "rr7", "rr14",
]
FEATURES_CAT: list[str] = ["centro_distribucion"]


def build_features(fc_rows: pd.DataFrame, daily_actuals: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Construye las features de cada pronóstico usando solo información conocida al emitirlo.

    Para una fila con emisión ``e`` y objetivo ``t = e + h`` solo se usa ``daily_actuals`` en fechas ≤ e−1.
    ``daily_actuals`` es (fecha × CD) y puede ser una vista truncada, como ocurre en producción.

    Familias de features: base (``lfc``, ``h``, CD), calendario y picos (``es_martes``, ``es_finde``,
    ``spike_lag7``, ``fc_util``), rolling (``rm7``, ``rm14``, ``rstd7``, ``fc_vs_rm7``, ``lag7_vs_rm14``) y
    error reciente del modelo base (``rr7``, ``rr14``). El target es ``y = log(real / fc)``.
    """
    if fc_rows.h.max() > 6:
        raise ValueError("lag7 requiere horizonte ≤ 6 para no mirar el futuro")
    rows = []
    for cd, g in fc_rows.groupby("centro_distribucion"):
        a = daily_actuals[cd].astype(float)
        full_idx = pd.date_range(a.index.min(), max(a.index.max(), g.fecha_despacho.max()), freq="D")
        a = a.reindex(full_idx)
        known7 = a.rolling(7, min_periods=3).mean().shift(1)  # el valor en e usa datos ≤ e−1
        known14 = a.rolling(14, min_periods=7).mean().shift(1)
        std7 = a.rolling(7, min_periods=5).std().shift(1)
        lag7 = a.shift(7)  # el valor en t usa t−7 ≤ e−1 si h ≤ 6
        known14_at_lag = a.rolling(14, min_periods=7).mean().shift(8)  # media conocida en la fecha del lag7
        gg = g.copy()
        e, t = gg.fecha_emision, gg.fecha_despacho
        gg["rm7"] = e.map(known7)
        gg["rm14"] = e.map(known14)
        gg["rstd7"] = e.map(std7)
        gg["lag7"] = t.map(lag7)
        gg["lag7_base"] = t.map(known14_at_lag)
        gg["capacidad"] = t.map(capacity[cd])
        rows.append(gg)
    f = pd.concat(rows).sort_values(["centro_distribucion", "h", "fecha_despacho"])

    # Ratio real/fc móvil por (CD, h): ventana sobre fechas objetivo, desplazada h+1 (conocido hasta e−1)
    f["_ratio"] = f["real"] / f["fc"]
    for w in (7, 14):
        f[f"rr{w}"] = np.nan
    for (_, h), g in f.groupby(["centro_distribucion", "h"]):
        s = g.set_index("fecha_despacho")["_ratio"]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        for w in (7, 14):
            r = s.rolling(w, min_periods=3).mean().shift(int(h) + 1)
            f.loc[g.index, f"rr{w}"] = g.fecha_despacho.map(r).to_numpy()

    f["dow"] = f.fecha_despacho.dt.dayofweek
    f["es_martes"] = (f.dow == 1).astype(float)
    f["es_finde"] = (f.dow >= 5).astype(float)
    f["lfc"] = np.log(f["fc"])
    f["fc_util"] = f["fc"] / f["capacidad"]
    f["fc_vs_rm7"] = f["fc"] / f["rm7"]
    f["lag7_vs_rm14"] = f["lag7"] / f["rm14"]
    f["spike_lag7"] = np.where(
        f["lag7"].notna() & f["lag7_base"].notna(),
        (f["lag7"] > PEAK_MULT * f["lag7_base"]).astype(float),
        np.nan,
    )
    f["y"] = np.log(f["real"] / f["fc"])
    return (
        f.drop(columns=["_ratio", "lag7_base"])
        .sort_values(["fecha_emision", "centro_distribucion", "h"])
        .reset_index(drop=True)
    )
