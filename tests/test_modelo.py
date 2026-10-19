from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desafio_logistica.features import build_features
from desafio_logistica.metrics import error_metrics
from desafio_logistica.models import rolling_origin_backtest


@pytest.fixture(scope="module")
def feat(tables) -> pd.DataFrame:
    return build_features(tables["fc"], tables["daily"], tables["cap_w"])


def test_sin_fuga_temporal(tables, feat) -> None:
    """Recalcular las features con la historia truncada a e−1 debe dar exactamente lo mismo."""
    daily, fc, cap_w = tables["daily"], tables["fc"], tables["cap_w"]
    cols = ["rm7", "rm14", "rstd7", "lag7", "rr7", "rr14", "spike_lag7"]
    rng = np.random.default_rng(42)
    for _, r in feat.iloc[rng.choice(len(feat), 25, replace=False)].iterrows():
        corte = r.fecha_emision - pd.Timedelta(days=1)
        sub = fc[fc.centro_distribucion == r.centro_distribucion].copy()
        sub.loc[sub.fecha_despacho > corte, "real"] = np.nan
        q = build_features(sub, daily[daily.index <= corte], cap_w)
        q = q[(q.fecha_emision == r.fecha_emision) & (q.h == r.h)].iloc[0]
        for c in cols:
            assert (pd.isna(r[c]) and pd.isna(q[c])) or np.isclose(r[c], q[c]), f"fuga en {c}"


def test_horizonte_mayor_a_6_se_rechaza(tables) -> None:
    fc = tables["fc"].copy()
    fc.loc[fc.index[0], "h"] = 7
    with pytest.raises(ValueError, match="horizonte"):
        build_features(fc, tables["daily"], tables["cap_w"])


def test_backtest_reproduce_el_notebook(tables, feat) -> None:
    bt, _ = rolling_origin_backtest(feat, t0=tables["daily"].index.min())
    assert bt.fold.nunique() == 5
    assert len(bt) == 708
    base = error_metrics(bt, None, pred="pred_base").WAPE.iloc[0]
    ridge = error_metrics(bt, None, pred="pred_ridge").WAPE.iloc[0]
    assert base == pytest.approx(0.2073, abs=1e-3)
    assert ridge == pytest.approx(0.0719, abs=3e-3)
    assert ridge < base  # el modelo supera al base
