from __future__ import annotations

import pandas as pd
import pytest

from desafio_logistica.metrics import error_metrics


def test_wape_y_bias_con_calculo_a_mano() -> None:
    df = pd.DataFrame({"real": [100.0, 200.0], "fc": [80.0, 250.0]})
    m = error_metrics(df).iloc[0]
    # |20| + |-50| = 70 sobre 300  ->  WAPE 7/30 ; (20 - 50) = -30 sobre 300 -> Bias -0.1
    assert m.WAPE == pytest.approx(70 / 300)
    assert m.Bias == pytest.approx(-30 / 300)
    assert m["%_filas_subestimadas"] == pytest.approx(0.5)


def test_signo_del_bias_positivo_es_subestimacion() -> None:
    df = pd.DataFrame({"cd": ["A", "A", "B", "B"], "real": [10.0, 10.0, 10.0, 10.0], "fc": [5.0, 5.0, 15.0, 15.0]})
    m = error_metrics(df, "cd")
    assert m.loc["A", "Bias"] > 0  # pronóstico bajo la demanda
    assert m.loc["B", "Bias"] < 0  # pronóstico sobre la demanda
