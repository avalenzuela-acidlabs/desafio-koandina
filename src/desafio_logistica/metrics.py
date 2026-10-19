"""Métricas de error del pronóstico (Tarea 1)."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def error_metrics(
    df: pd.DataFrame,
    by: Sequence[str] | str | None = None,
    real: str = "real",
    pred: str = "fc",
) -> pd.DataFrame:
    """WAPE y Bias agregados (no promedios de ratios).

    - ``WAPE = Σ|real − pred| / Σ|real|``
    - ``Bias = Σ(real − pred) / Σ|real|``: positivo = subestimación, negativo = sobreestimación.

    También devuelve ``%_filas_subestimadas``, ``n`` y ``WAPE_aleatorio`` (WAPE − |Bias|, la parte del
    error que no se explica por un sesgo sistemático). Con ``by=None`` entrega una sola fila ``GLOBAL``.
    """
    t = df.assign(
        _abs=(df[real] - df[pred]).abs(),
        _err=df[real] - df[pred],
        _den=df[real].abs(),
        _under=(df[real] > df[pred]).astype(float),
    )
    if by is None:
        t["_all"] = "GLOBAL"
        by = ["_all"]
    g = t.groupby(by, observed=True)
    s = g[["_abs", "_err", "_den", "_under"]].agg({"_abs": "sum", "_err": "sum", "_den": "sum", "_under": "mean"})
    out = pd.DataFrame(
        {
            "WAPE": s._abs / s._den,
            "Bias": s._err / s._den,
            "%_filas_subestimadas": s._under,
            "n": g.size(),
        }
    )
    out["WAPE_aleatorio"] = out.WAPE - out.Bias.abs()
    return out
