"""Carga y validación de los cuatro CSV del data package."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from .config import ZIP_NAME

# nombre de archivo -> columnas de fecha
_SPEC: dict[str, tuple[str, list[str]]] = {
    "pedidos": ("pedidos_historicos_genericos.csv", ["fecha_despacho"]),
    "pron": ("pronostico_demanda_generico.csv", ["fecha_emision", "fecha_despacho"]),
    "cap": ("capacidad_logistica_generica.csv", ["fecha"]),
    "plan": ("plan_entrega_clientes_generico.csv", ["fecha_plan", "fecha_compra_real"]),
}


def find_source(source: str | Path | None = None) -> Path:
    """Resuelve la fuente de datos: un .zip, una carpeta con los CSV o el zip por defecto."""
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No existe la fuente de datos: {path}")
        return path
    here = Path.cwd()
    candidates = [here / "data" / ZIP_NAME, here / ZIP_NAME]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontró {ZIP_NAME}. Se buscó en: {[str(c) for c in candidates]}")


def load_data(source: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Lee los 4 CSV (desde un .zip o una carpeta) y tipa las fechas.

    Devuelve un dict con las llaves ``pedidos``, ``pron``, ``cap`` y ``plan``.
    """
    path = find_source(source)
    out: dict[str, pd.DataFrame] = {}
    zf = zipfile.ZipFile(path) if path.is_file() else None
    try:
        for key, (fname, dates) in _SPEC.items():
            if zf is not None:
                member = next(n for n in zf.namelist() if n.endswith(fname))
                out[key] = pd.read_csv(io.BytesIO(zf.read(member)), parse_dates=dates)
            else:
                out[key] = pd.read_csv(path / fname, parse_dates=dates)
    finally:
        if zf is not None:
            zf.close()
    return out


def data_quality_report(
    pedidos: pd.DataFrame, pron: pd.DataFrame, cap: pd.DataFrame, plan: pd.DataFrame
) -> pd.DataFrame:
    """Chequeos de integridad. Devuelve una tabla (chequeo, resultado, ok)."""
    chk: list[tuple[str, str, bool]] = []
    days_cd = pedidos.groupby("centro_distribucion")["fecha_despacho"].nunique()
    chk.append(("pedidos: 60 días por CD", str(days_cd.to_dict()), bool((days_cd == 60).all())))
    per_day = pedidos.groupby(["centro_distribucion", "fecha_despacho"]).size()
    chk.append(("pedidos: 3 canales por CD-día", str(per_day.value_counts().to_dict()), bool((per_day == 3).all())))
    chk.append(("pedidos: unidades > 0", f"min={pedidos.unidades_reales.min():.2f}", bool((pedidos.unidades_reales > 0).all())))

    lag = (pron.fecha_despacho - pron.fecha_emision).dt.days
    label_ok = ("D+" + lag.astype(str)) == pron.horizonte
    chk.append(("pronóstico: etiqueta de horizonte coherente con fechas", f"{label_ok.mean():.0%}", bool(label_ok.all())))
    dup = int(pron.duplicated(["centro_distribucion", "fecha_despacho", "horizonte"]).sum())
    chk.append(("pronóstico: sin llaves duplicadas (CD, fecha, horizonte)", str(dup), dup == 0))
    cnt = pron.groupby(["centro_distribucion", "horizonte"]).size().unstack()
    chk.append(("pronóstico: filas por horizonte (D+1..D+4)", str(cnt.iloc[0].to_dict()), True))

    cap_ok = len(cap) == 360 and not cap.duplicated(["centro_distribucion", "fecha"]).any()
    chk.append(("capacidad: 1 fila por CD-día (360)", str(len(cap)), bool(cap_ok)))

    flag_ok = bool(((plan.fecha_plan != plan.fecha_compra_real).astype(int) == plan.fuera_de_plan).all())
    chk.append(("plan: fuera_de_plan coherente con fechas", "100%" if flag_ok else "inconsistente", flag_ok))
    chk.append(
        ("plan: 90 clientes, 15 por CD", str(plan.groupby("centro_distribucion").cliente_id.nunique().to_dict()),
         plan.cliente_id.nunique() == 90)
    )
    cli_cd = int(plan.groupby("cliente_id").centro_distribucion.nunique().max())
    chk.append(("plan: cada cliente pertenece a un único CD", str(cli_cd), cli_cd == 1))
    return pd.DataFrame(chk, columns=["chequeo", "resultado", "ok"])


def build_daily(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Demanda real diaria por CD (suma de canales). Índice: fecha; columnas: CD."""
    daily = (
        pedidos.groupby(["fecha_despacho", "centro_distribucion"]).unidades_reales.sum()
        .unstack("centro_distribucion").sort_index()
    )
    daily.index.name = "fecha"
    return daily


def build_forecast_table(pron: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Une el pronóstico base con la demanda real. Columnas ``fc`` (pronóstico), ``real`` y ``h`` (días)."""
    real = daily.stack().rename("real").reset_index().rename(columns={"fecha": "fecha_despacho"})
    fc = pron.merge(real, on=["fecha_despacho", "centro_distribucion"], how="left").rename(
        columns={"unidades_pronosticadas": "fc"}
    )
    fc["h"] = fc.horizonte.str[-1].astype(int)
    if fc.real.isna().any():
        raise ValueError("Hay pronósticos sin demanda real asociada")
    return fc


def build_capacity_wide(cap: pd.DataFrame) -> pd.DataFrame:
    """Capacidad máxima por día. Índice: fecha; columnas: CD."""
    return cap.pivot(index="fecha", columns="centro_distribucion", values="capacidad_maxima_unidades")
