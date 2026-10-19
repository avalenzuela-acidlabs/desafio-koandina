"""Línea de comandos: corre las 5 tareas sobre el data package y guarda las tablas en CSV."""
from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import HORIZONS, MIN_HISTORY_DAYS
from .data import build_capacity_wide, build_daily, build_forecast_table, data_quality_report, load_data
from .features import build_features
from .flatten import simulate_alphas
from .gate import gated_forecast, threshold_sweep
from .impact import impact_table, relative_reduction
from .metrics import error_metrics
from .models import daily_origin_backtest, rolling_origin_backtest
from .plan import (
    bias_by_group,
    client_metrics,
    date_bias,
    deviation_concentration,
    out_of_plan_summary,
    prepare_plan,
    volume_bias,
    weekday_shares,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def _save(df: pd.DataFrame, out_dir: Path | None, name: str) -> None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{name}.csv")


def run(data: str | Path | None, out_dir: Path | None, as_of: str | None) -> None:
    """Ejecuta el análisis completo e imprime un resumen de cada tarea."""
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    d = load_data(data)
    pedidos, pron, cap, plan = d["pedidos"], d["pron"], d["cap"], d["plan"]
    daily = build_daily(pedidos)
    cap_w = build_capacity_wide(cap)
    fc = build_forecast_table(pron, daily)
    cds = sorted(daily.columns)

    _section("0. Calidad de datos")
    quality = data_quality_report(pedidos, pron, cap, plan)
    print(quality.to_string(index=False))
    _save(quality, out_dir, "00_calidad_datos")

    _section("1. Diagnóstico del pronóstico base")
    m_cd, m_h = error_metrics(fc, "centro_distribucion"), error_metrics(fc, "horizonte")
    print("Global:\n", error_metrics(fc))
    print("\nPor CD:\n", m_cd.sort_values("Bias", ascending=False))
    print("\nPor horizonte:\n", m_h)
    _save(m_cd, out_dir, "01_metricas_por_cd")
    _save(m_h, out_dir, "01_metricas_por_horizonte")
    _save(error_metrics(fc, ["centro_distribucion", "horizonte"]), out_dir, "01_metricas_cd_horizonte")

    _section("2. Plan de entrega")
    pl = prepare_plan(plan)
    vb, db, oop = volume_bias(pl), date_bias(pl), out_of_plan_summary(pl)
    print(f"Sesgo de volumen del plan: {vb['bias']:+.2%} (real/plan ≈ {vb['factor_real_sobre_plan']:.3f}, p={vb['p_ttest']:.1e})")
    print(f"Desfase medio de fecha (fuera de plan): {db['desfase_medio']:+.2f} días; atrasadas {db['pct_atrasadas']:.1%}")
    print(f"Compras fuera de plan: {oop['pct_total']:.1%} ({oop['n_fuera']}/{oop['n']}), {oop['pct_volumen']:.1%} del volumen")
    print("\nPor CD:\n", oop["por_cd"])
    print("\nPor día planificado:\n", oop["por_dia"])
    cli = client_metrics(pl)
    conc = deviation_concentration(cli)
    print(f"\nAdhesión media por cliente: {cli.adhesion.mean():.1%}; top 10 % de clientes = {conc['top10']:.0%} "
          f"de los desvíos (Gini {conc['gini']:.2f})")
    shares = weekday_shares(pl)
    print("\nUnidades por día, plan vs real:\n", shares)
    _save(bias_by_group(pl, "centro_distribucion"), out_dir, "02_sesgo_plan_por_cd")
    _save(oop["por_cd"], out_dir, "02_fuera_de_plan_por_cd")
    _save(shares, out_dir, "02_unidades_por_dia")
    _save(cli.set_index("cliente_id"), out_dir, "02_metricas_por_cliente")
    sim = simulate_alphas(daily[cds], cap_w[cds])
    print("\nSimulación de aplanamiento (α = fracción de carga excedente que acepta moverse):\n", sim)
    _save(sim, out_dir, "02_simulacion_aplanamiento")

    _section("3. Modelo de corrección (backtest de origen rodante)")
    feat = build_features(fc, daily, cap_w)
    bt, _ = rolling_origin_backtest(feat, t0=daily.index.min())
    labels = {"pred_base": "Base", "pred_recal_cd": "Recalibración por CD", "pred_ridge": "Ridge (propuesto)",
              "pred_hgb": "HistGradientBoosting"}
    res = pd.concat({lbl: error_metrics(bt, None, pred=col).iloc[0] for col, lbl in labels.items()}, axis=1).T
    res["mejora_WAPE_pp"] = (res.WAPE.iloc[0] - res.WAPE) * 100
    print(f"Folds: {bt.fold.nunique()}, filas evaluadas: {len(bt)}")
    print(res[["WAPE", "Bias", "mejora_WAPE_pp", "n"]])
    by_h = pd.concat({lbl: error_metrics(bt, "horizonte", pred=c).WAPE for c, lbl in labels.items()}, axis=1)
    print("\nWAPE por horizonte:\n", by_h.reindex(HORIZONS))
    _save(res, out_dir, "03_wape_backtest")
    _save(bt, out_dir, "03_predicciones_backtest")

    _section(f"4. Gate de historia mínima (umbral = {MIN_HISTORY_DAYS} días)")
    cutoff = pd.Timestamp(as_of) if as_of else pron.fecha_emision.max()
    gated, gate_tbl = gated_forecast(fc, daily, cap_w, cutoff)
    print(f"Fecha de corte: {cutoff.date()}")
    print(gate_tbl)
    sweep = threshold_sweep(daily_origin_backtest(feat, t0=daily.index.min()))
    print("\nWAPE global según el umbral T:\n", sweep)
    _save(gate_tbl, out_dir, "04_gate_decisiones")
    _save(sweep, out_dir, "04_umbral_gate")

    _section("5. Impacto en el negocio (flete de emergencia = 1,8× la flota planificada)")
    imp = impact_table(bt, cap_w)
    red = relative_reduction(imp)
    print(imp, "\n\nVariación Base → Corregido:\n", red)
    _save(imp, out_dir, "05_impacto")
    _save(red, out_dir, "05_variacion_relativa")

    if out_dir is not None:
        print(f"\nTablas guardadas en {out_dir.resolve()}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="desafio_logistica", description=__doc__)
    parser.add_argument("--data", help="Zip o carpeta con los 4 CSV (por defecto: data/data_package_desafio_logistica.zip)")
    parser.add_argument("--out", type=Path, default=None, help="Carpeta donde guardar las tablas en CSV")
    parser.add_argument(
        "--as-of", help="Fecha de corte del Gate (YYYY-MM-DD); por defecto, la última fecha de emisión del pronóstico"
    )
    args = parser.parse_args(argv)
    run(args.data, args.out, args.as_of)


if __name__ == "__main__":
    main()
