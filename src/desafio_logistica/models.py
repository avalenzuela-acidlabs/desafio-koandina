"""Modelos de corrección del pronóstico y backtests de origen rodante (Tarea 3)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RATIO_BOUNDS, RIDGE_ALPHA, SEED
from .features import FEATURES_CAT, FEATURES_NUM

# Ablación: qué familias de features aportan al Ridge
ABLATIONS: dict[str, tuple[list[str], list[str]]] = {
    "ridge_solo_nivel": (["lfc", "h"], FEATURES_CAT),
    "ridge_+calendario": (["lfc", "h", "es_martes", "es_finde", "fc_util"], FEATURES_CAT),
    "ridge_+rolling": (
        ["lfc", "h", "es_martes", "es_finde", "fc_util", "rm7", "rm14", "rstd7", "fc_vs_rm7", "lag7_vs_rm14", "spike_lag7"],
        FEATURES_CAT,
    ),
}


def ridge_with(num: list[str], cat: list[str], alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Ridge con imputación por mediana y estandarización de las numéricas, y one-hot para las categóricas."""
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
    return Pipeline([("pre", pre), ("m", Ridge(alpha=alpha))])


def make_ridge(alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Modelo propuesto: Ridge sobre ``log(real / pronóstico base)`` con todas las features."""
    return ridge_with(FEATURES_NUM, FEATURES_CAT, alpha)


def make_hgb() -> Pipeline:
    """Challenger no lineal (intercambiable por LightGBM)."""
    pre = ColumnTransformer(
        [("num", "passthrough", FEATURES_NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)]
    )
    model = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=15, random_state=SEED
    )
    return Pipeline([("pre", pre), ("m", model)])


MODELS: dict[str, Callable[[], Pipeline]] = {"ridge": make_ridge, "hgb": make_hgb}


def apply_correction(fc: pd.Series, log_ratio_hat: np.ndarray) -> pd.Series:
    """Pronóstico corregido ``fc · exp(ŷ)``, con el ratio acotado a ``RATIO_BOUNDS``."""
    lo, hi = np.log(RATIO_BOUNDS[0]), np.log(RATIO_BOUNDS[1])
    return fc * np.exp(np.clip(log_ratio_hat, lo, hi))


def rolling_origin_backtest(
    f: pd.DataFrame,
    t0: pd.Timestamp,
    initial_days: int = 28,
    step_days: int = 7,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Backtest de origen rodante con ventana expansiva.

    En cada origen ``E0`` entrena con las filas cuyo target ya ocurrió (``fecha_despacho < E0``) y predice las
    emisiones en ``[E0, E0 + step)``. Se repite avanzando ``step_days`` hasta agotar los datos.

    ``t0`` es el primer día de la historia. Devuelve las predicciones fuera de muestra (``pred_base``,
    ``pred_recal_cd``, ``pred_ridge``, ``pred_hgb`` y las columnas de ablación) y los coeficientes del Ridge por fold.
    """
    emis = (f.fecha_emision - t0).dt.days
    out: list[pd.DataFrame] = []
    coefs: list[pd.DataFrame] = []
    e0, fold = initial_days, 0
    while e0 <= emis.max():
        tr = f[f.fecha_despacho < t0 + pd.Timedelta(days=e0)]
        te = f[(emis >= e0) & (emis < e0 + step_days)].copy()
        if te.empty:
            break
        te["fold"] = fold
        te["origen"] = t0 + pd.Timedelta(days=e0)
        te["n_train"] = len(tr)
        # (a) recalibración simple por CD: razón histórica Σreal / Σfc
        ratio = tr.groupby("centro_distribucion").apply(lambda g: g.real.sum() / g.fc.sum(), include_groups=False)
        te["pred_recal_cd"] = te.fc * te.centro_distribucion.map(ratio).fillna(1.0)
        # (b)(c) modelos de corrección
        for name, factory in MODELS.items():
            mdl = factory().fit(tr[FEATURES_NUM + FEATURES_CAT], tr.y)
            te[f"pred_{name}"] = apply_correction(te.fc, mdl.predict(te[FEATURES_NUM + FEATURES_CAT]))
            if name == "ridge":
                names = mdl.named_steps["pre"].get_feature_names_out()
                coefs.append(pd.DataFrame({"feature": names, "coef": mdl.named_steps["m"].coef_, "fold": fold}))
        for name, (num, cat) in ABLATIONS.items():
            mdl = ridge_with(num, cat).fit(tr[num + cat], tr.y)
            te[f"pred_{name}"] = apply_correction(te.fc, mdl.predict(te[num + cat]))
        out.append(te)
        e0 += step_days
        fold += 1
    bt = pd.concat(out, ignore_index=True)
    bt["pred_base"] = bt["fc"]
    return bt, coefs


def daily_origin_backtest(f: pd.DataFrame, t0: pd.Timestamp, first_origin: int = 8) -> pd.DataFrame:
    """Reentrena cada día con todo lo conocido y predice las emisiones de ese día.

    Agrega ``hist_days`` (días de historia disponibles) para evaluar umbrales del Gate.
    """
    emis = (f.fecha_emision - t0).dt.days
    res = []
    for e0 in range(first_origin, int(emis.max()) + 1):
        tr = f[f.fecha_despacho < t0 + pd.Timedelta(days=e0)]
        te = f[emis == e0].copy()
        if len(tr) < 25 or te.empty:
            continue
        mdl = make_ridge().fit(tr[FEATURES_NUM + FEATURES_CAT], tr.y)
        te["pred_model"] = apply_correction(te.fc, mdl.predict(te[FEATURES_NUM + FEATURES_CAT]))
        te["hist_days"] = e0
        res.append(te)
    return pd.concat(res, ignore_index=True)
