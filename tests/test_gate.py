from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desafio_logistica.gate import continuous_history_days, evaluate_history_gate, gated_forecast

AS_OF = pd.Timestamp("2026-02-20")


def _serie(dias: int, fin: pd.Timestamp | None = None) -> pd.Series:
    idx = pd.date_range(end=fin or AS_OF - pd.Timedelta(days=1), periods=dias, freq="D")
    return pd.Series(100.0, index=idx)


@pytest.mark.parametrize(("dias", "estado"), [(44, "FALLBACK"), (45, "TRAIN"), (46, "TRAIN"), (20, "FALLBACK")])
def test_umbral_de_45_dias(dias: int, estado: str) -> None:
    daily = pd.DataFrame({"CD_X": _serie(dias)})
    assert evaluate_history_gate(daily, ["CD_X"], AS_OF)["CD_X"].estado == estado


def test_un_hueco_reinicia_la_cuenta() -> None:
    s = _serie(60)
    s.loc["2026-02-10":"2026-02-12"] = np.nan
    n, motivo = continuous_history_days(s, AS_OF)
    assert n == 7  # del 13 al 19 de febrero
    assert "hueco_detectado" in motivo


def test_serie_desactualizada_cuenta_cero() -> None:
    n, motivo = continuous_history_days(_serie(60, fin=AS_OF - pd.Timedelta(days=5)), AS_OF)
    assert n == 0
    assert "desactualizada" in motivo


@pytest.mark.parametrize("entrada", [None, pd.DataFrame(), pd.DataFrame({"CD_X": [np.nan, np.nan]})])
def test_entradas_patologicas_no_lanzan_y_caen_a_fallback(entrada: pd.DataFrame | None) -> None:
    dec = evaluate_history_gate(entrada, ["CD_X"], AS_OF)
    assert dec["CD_X"].estado == "FALLBACK"


def test_cd_desconocido_cae_a_fallback() -> None:
    daily = pd.DataFrame({"CD_X": _serie(60)})
    assert evaluate_history_gate(daily, ["CD_FANTASMA"], AS_OF)["CD_FANTASMA"].estado == "FALLBACK"


def test_gated_forecast_usa_modelo_solo_en_cds_elegibles(tables) -> None:
    daily, fc, cap_w = tables["daily"], tables["fc"], tables["cap_w"]
    hist = daily.copy()
    hist.loc[hist.index < "2026-02-01", "CD_OESTE"] = np.nan  # ~19 días de historia
    out, tabla = gated_forecast(fc, hist, cap_w, AS_OF)
    assert tabla.loc["CD_OESTE", "estado"] == "FALLBACK"
    assert tabla.loc["CD_NORTE", "estado"] == "TRAIN"
    futuros = out[out.fecha_emision >= AS_OF]
    fuente = futuros.groupby("centro_distribucion").fuente.first()
    assert fuente["CD_OESTE"] == "FALLBACK_BASE"
    assert fuente["CD_NORTE"] == "MODELO_CORREGIDO"
    # en fallback el pronóstico final es exactamente el base
    oeste = futuros[futuros.centro_distribucion == "CD_OESTE"]
    assert np.allclose(oeste.fc_final, oeste.fc)
    assert np.isfinite(out.fc_final).all()


def test_gated_forecast_sin_datos_no_lanza(tables) -> None:
    out, tabla = gated_forecast(tables["fc"], None, tables["cap_w"], AS_OF)
    assert (tabla.estado == "FALLBACK").all()
    assert (out.fuente == "FALLBACK_BASE").all()
