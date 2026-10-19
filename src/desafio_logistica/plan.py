"""Análisis del plan de entrega de clientes (Tarea 2)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import DOW_ES


def prepare_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Agrega día de semana, desfases de fecha y de volumen, y el log-ratio real/plan."""
    pl = plan.copy()
    pl["dow_plan"] = pl.fecha_plan.dt.dayofweek
    pl["dow_real"] = pl.fecha_compra_real.dt.dayofweek
    pl["delta_dias"] = (pl.fecha_compra_real - pl.fecha_plan).dt.days
    pl["delta_uni"] = pl.unidades_reales - pl.unidades_plan
    pl["log_ratio"] = np.log(pl.unidades_reales / pl.unidades_plan)
    return pl


def volume_bias(pl: pd.DataFrame) -> dict[str, Any]:
    """Sesgo de volumen del plan: ``Σ(real − plan) / Σreal`` (positivo = el plan se queda corto)."""
    n = len(pl)
    n_pos = int((pl.delta_uni > 0).sum())
    ci = stats.t.interval(0.95, n - 1, loc=pl.log_ratio.mean(), scale=stats.sem(pl.log_ratio))
    return {
        "unid_plan": float(pl.unidades_plan.sum()),
        "unid_real": float(pl.unidades_reales.sum()),
        "bias": float(pl.delta_uni.sum() / pl.unidades_reales.sum()),
        "compras_real_mayor_plan": n_pos,
        "n": n,
        "p_signos": float(stats.binomtest(n_pos, n, 0.5).pvalue),
        "log_ratio_medio": float(pl.log_ratio.mean()),
        "ic95_log_ratio": (float(ci[0]), float(ci[1])),
        "factor_real_sobre_plan": math.exp(float(pl.log_ratio.mean())),
        "p_ttest": float(stats.ttest_1samp(pl.log_ratio, 0.0).pvalue),
    }


def bias_by_group(pl: pd.DataFrame, col: str) -> pd.DataFrame:
    """Sesgo de volumen del plan y % de compras fuera de plan por grupo."""
    g = pl.groupby(col).agg(
        unid_plan=("unidades_plan", "sum"),
        unid_real=("unidades_reales", "sum"),
        delta=("delta_uni", "sum"),
        pct_fuera_plan=("fuera_de_plan", "mean"),
        n=("fuera_de_plan", "size"),
    )
    g["Bias_vol"] = g.delta / g.unid_real
    return g.drop(columns="delta")


def date_bias(pl: pd.DataFrame) -> dict[str, Any]:
    """Sesgo de fecha, calculado solo sobre las compras fuera de plan."""
    oop = pl[pl.fuera_de_plan == 1]
    late = int((oop.delta_dias > 0).sum())
    return {
        "n_fuera_plan": len(oop),
        "desfase_medio": float(oop.delta_dias.mean()),
        "desfase_mediana": float(oop.delta_dias.median()),
        "desfase_abs_medio": float(oop.delta_dias.abs().mean()),
        "pct_atrasadas": float((oop.delta_dias > 0).mean()),
        "pct_adelantadas": float((oop.delta_dias < 0).mean()),
        "pct_dentro_3_dias": float((oop.delta_dias.abs() <= 3).mean()),
        "p_signos": float(stats.binomtest(late, len(oop), 0.5).pvalue),
    }


def out_of_plan_summary(pl: pd.DataFrame) -> dict[str, Any]:
    """% de compras fuera de plan: total, por CD y por día de la semana planificado (con test χ²)."""
    by_cd = pl.groupby("centro_distribucion").agg(
        pct_fuera=("fuera_de_plan", "mean"),
        n=("fuera_de_plan", "size"),
        desfase_medio=("delta_dias", lambda s: s[s != 0].mean()),
    ).sort_values("pct_fuera", ascending=False)
    by_dow = pl.groupby("dow_plan").agg(
        pct_fuera=("fuera_de_plan", "mean"), n=("fuera_de_plan", "size"), unid_plan=("unidades_plan", "sum")
    )
    by_dow.index = [DOW_ES[i] for i in by_dow.index]
    oop = pl[pl.fuera_de_plan == 1]
    return {
        "pct_total": float(pl.fuera_de_plan.mean()),
        "n_fuera": int(pl.fuera_de_plan.sum()),
        "n": len(pl),
        "pct_volumen": float(oop.unidades_reales.sum() / pl.unidades_reales.sum()),
        "por_cd": by_cd,
        "por_dia": by_dow,
        "p_chi2_cd": float(stats.chi2_contingency(pd.crosstab(pl.centro_distribucion, pl.fuera_de_plan)).pvalue),
        "p_chi2_dia": float(stats.chi2_contingency(pd.crosstab(pl.dow_plan, pl.fuera_de_plan)).pvalue),
    }


def client_metrics(pl: pd.DataFrame) -> pd.DataFrame:
    """Métricas por cliente: adhesión al plan, desfase absoluto medio, ratio de volumen y unidades fuera de plan."""
    cli = pl.groupby(["cliente_id", "centro_distribucion"]).agg(
        entregas=("fuera_de_plan", "size"),
        adhesion=("fuera_de_plan", lambda s: 1 - s.mean()),
        desfase_medio_abs=("delta_dias", lambda s: s.abs().mean()),
        ratio_vol=("unidades_reales", "sum"),
        plan_vol=("unidades_plan", "sum"),
        unid_fuera=("unidades_reales", lambda s: s[pl.loc[s.index, "fuera_de_plan"] == 1].sum()),
    ).reset_index()
    cli["ratio_vol"] = cli.ratio_vol / cli.plan_vol
    return cli


def deviation_concentration(cli: pd.DataFrame) -> dict[str, Any]:
    """Concentración de los desvíos: % de unidades fuera de plan que explican el top 10 % y 20 % de clientes, y Gini."""
    srt = cli.sort_values("unid_fuera", ascending=False)
    cum = srt.unid_fuera.cumsum() / srt.unid_fuera.sum()
    n = len(cli)
    k10, k20 = max(1, round(0.10 * n)), max(1, round(0.20 * n))
    x = np.sort(cli.unid_fuera.to_numpy())
    gini = (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n
    return {"top10": float(cum.iloc[k10 - 1]), "top20": float(cum.iloc[k20 - 1]), "gini": float(gini), "curva": cum}


def migration_matrix(pl: pd.DataFrame) -> pd.DataFrame:
    """Matriz día planificado (filas) → día real (columnas)."""
    mig = pd.crosstab(pl.dow_plan.map(dict(enumerate(DOW_ES))), pl.dow_real.map(dict(enumerate(DOW_ES))))
    return mig.reindex(index=DOW_ES[:5], columns=DOW_ES).fillna(0).astype(int)


def weekday_shares(pl: pd.DataFrame) -> pd.DataFrame:
    """% de las unidades por día de semana: planificado vs real."""
    share_plan = pl.groupby("dow_plan").unidades_plan.sum() / pl.unidades_plan.sum()
    share_real = pl.groupby("dow_real").unidades_reales.sum() / pl.unidades_reales.sum()
    out = pd.DataFrame({"%unid_plan": share_plan, "%unid_real": share_real}).reindex(range(7)).fillna(0)
    out.index = DOW_ES
    out["Δ pp"] = (out["%unid_real"] - out["%unid_plan"]) * 100
    return out
