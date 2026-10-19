"""Gate de historia mínima / Cold Start (Tarea 4)."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MIN_HISTORY_DAYS, RATIO_BOUNDS
from .features import FEATURES_CAT, FEATURES_NUM, build_features
from .metrics import error_metrics
from .models import make_ridge

MIN_TRAIN_ROWS = 40  # filas mínimas para entrenar; con menos se cae a fallback


@dataclass(frozen=True)
class GateDecision:
    """Resultado de evaluar la cobertura histórica de un CD."""

    centro_distribucion: str
    dias_continuos: int
    umbral: int
    estado: str  # "TRAIN" | "FALLBACK"
    razon: str


def continuous_history_days(series: pd.Series | None, as_of: pd.Timestamp) -> tuple[int, str]:
    """Días consecutivos con dato (> 0 y no NaN) que terminan en ``as_of − 1``. Devuelve ``(días, motivo)``.

    Un hueco reinicia el conteo y una serie desactualizada (último dato anterior a ``as_of − 1``) cuenta 0.
    """
    if series is None or len(series) == 0:
        return 0, "sin_historia"
    s = pd.Series(series).dropna()
    s = s[(s.index < as_of) & (s > 0)]
    if s.empty:
        return 0, "sin_historia"
    last = s.index.max()
    if last < as_of - pd.Timedelta(days=1):
        return 0, f"serie_desactualizada(último_dato={last.date()})"
    days = pd.date_range(s.index.min(), last, freq="D")
    present = pd.Series(days.isin(s.index), index=days)
    gaps = present[~present]
    if gaps.empty:
        return int(len(days)), "continua"
    start = gaps.index.max() + pd.Timedelta(days=1)
    return int((last - start).days + 1), f"hueco_detectado(último_hueco={gaps.index.max().date()})"


def evaluate_history_gate(
    daily_actuals: pd.DataFrame | None,
    centers: Iterable[str],
    as_of: pd.Timestamp,
    min_days: int = MIN_HISTORY_DAYS,
) -> dict[str, GateDecision]:
    """Evalúa el Gate por CD. Es seguro ante cualquier entrada: nunca propaga excepciones."""
    decisions: dict[str, GateDecision] = {}
    for cd in centers:
        try:
            series = None
            if daily_actuals is not None and cd in getattr(daily_actuals, "columns", []):
                series = daily_actuals[cd]
            n, why = continuous_history_days(series, pd.Timestamp(as_of))
            ok = n >= min_days
            decisions[cd] = GateDecision(
                cd, n, min_days, "TRAIN" if ok else "FALLBACK",
                "cobertura_suficiente" if ok else f"historia_insuficiente[{why}]",
            )
        except Exception as exc:  # noqa: BLE001 — contrato: el Gate nunca lanza
            decisions[cd] = GateDecision(cd, 0, min_days, "FALLBACK", f"error_controlado[{type(exc).__name__}]")
    return decisions


def gated_forecast(
    fc_rows: pd.DataFrame,
    daily_actuals: pd.DataFrame | None,
    capacity: pd.DataFrame,
    as_of: pd.Timestamp,
    min_days: int = MIN_HISTORY_DAYS,
    cap_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo con Gate: entrena solo con los CDs elegibles y aplica fallback al resto.

    Devuelve ``(pronósticos con columnas fc_final y fuente, tabla de decisiones del Gate)``. ``fuente`` es
    ``MODELO_CORREGIDO`` o ``FALLBACK_BASE``. Con ``cap_fallback`` el fallback se acota a la capacidad del CD.
    """
    centers = sorted(fc_rows.centro_distribucion.unique())
    dec = evaluate_history_gate(daily_actuals, centers, as_of, min_days)
    out = fc_rows.copy()
    out["fc_final"] = out["fc"].astype(float)
    out["fuente"] = "FALLBACK_BASE"
    eligible = [c for c, d in dec.items() if d.estado == "TRAIN"]
    if eligible and daily_actuals is not None:
        try:
            hist = daily_actuals.loc[daily_actuals.index < as_of, eligible]
            tr_rows = fc_rows[fc_rows.centro_distribucion.isin(eligible) & (fc_rows.fecha_despacho < as_of)].copy()
            tr_rows["real"] = [
                hist.at[d, c] if d in hist.index else np.nan
                for d, c in zip(tr_rows.fecha_despacho, tr_rows.centro_distribucion, strict=True)
            ]
            tr_f = build_features(tr_rows.dropna(subset=["real"]), hist, capacity).dropna(subset=["y"])
            tgt = fc_rows[fc_rows.centro_distribucion.isin(eligible) & (fc_rows.fecha_emision >= as_of)].copy()
            if len(tr_f) >= MIN_TRAIN_ROWS and not tgt.empty:  # sin pronósticos pendientes no hay nada que corregir
                mdl = make_ridge().fit(tr_f[FEATURES_NUM + FEATURES_CAT], tr_f.y)
                tgt["real"] = np.nan
                tf = build_features(tgt, hist, capacity)  # features de inferencia: solo historia < as_of
                yhat = np.clip(
                    mdl.predict(tf[FEATURES_NUM + FEATURES_CAT]), np.log(RATIO_BOUNDS[0]), np.log(RATIO_BOUNDS[1])
                )
                key = list(zip(tf.centro_distribucion, tf.fecha_emision, tf.fecha_despacho, strict=True))
                upd = dict(zip(key, tf.fc * np.exp(yhat), strict=True))
                idx = list(zip(out.centro_distribucion, out.fecha_emision, out.fecha_despacho, strict=True))
                mask = pd.Series([k in upd for k in idx], index=out.index)
                out.loc[mask, "fc_final"] = [upd[k] for k, m in zip(idx, mask, strict=True) if m]
                out.loc[mask, "fuente"] = "MODELO_CORREGIDO"
            elif len(tr_f) < MIN_TRAIN_ROWS:
                for c in eligible:
                    dec[c] = GateDecision(c, dec[c].dias_continuos, min_days, "FALLBACK", "filas_de_entrenamiento_insuficientes")
        except Exception as exc:  # noqa: BLE001 — resguardo final: ante cualquier fallo, pronóstico base
            for c in eligible:
                razon = f"error_entrenamiento[{type(exc).__name__}]"
                dec[c] = GateDecision(c, dec[c].dias_continuos, min_days, "FALLBACK", razon)
            out["fc_final"] = out["fc"].astype(float)
            out["fuente"] = "FALLBACK_BASE"
    if cap_fallback:
        c_map = capacity.stack().rename("cap").reset_index().rename(columns={"fecha": "fecha_despacho"})
        out = out.merge(c_map, on=["fecha_despacho", "centro_distribucion"], how="left")
        out["fc_final"] = np.where(out.fuente == "FALLBACK_BASE", np.minimum(out.fc_final, out.cap.fillna(np.inf)), out.fc_final)
        out = out.drop(columns="cap")
    table = pd.DataFrame([d.__dict__ for d in dec.values()]).set_index("centro_distribucion")
    return out, table


def threshold_sweep(dbt: pd.DataFrame, thresholds: Iterable[int] = (7, 14, 21, 28, 35, 45, 52)) -> pd.DataFrame:
    """WAPE global según el umbral T del Gate, sobre el backtest diario (``daily_origin_backtest``).

    Con historia ≥ T se usa el modelo; con menos, el pronóstico base. Muestra el costo de esperar más días.
    """
    rows = []
    for t in thresholds:
        pred = np.where(dbt.hist_days >= t, dbt.pred_model, dbt.fc)
        m = error_metrics(dbt.assign(pred=pred), None, pred="pred").iloc[0]
        rows.append(
            {"umbral_T": t, "%días_en_fallback": 1 - (dbt.hist_days >= t).mean(),
             "WAPE_global": m.WAPE, "Bias_global": m.Bias}
        )
    sweep = pd.DataFrame(rows).set_index("umbral_T")
    base = error_metrics(dbt, None, pred="fc").iloc[0]
    sweep["ganancia_vs_base_pp"] = (base.WAPE - sweep.WAPE_global) * 100
    sweep["costo_de_esperar_vs_T7_pp"] = (sweep.WAPE_global - sweep.WAPE_global.iloc[0]) * 100
    return sweep
