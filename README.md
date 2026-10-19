# Desafío Koandina · Optimización logística

Solución al desafío técnico de Data Scientist Lead: auditar el pronóstico de demanda de 6 centros de distribución (CDs), corregirlo con un modelo simple y proponer cómo ponerlo en producción en AWS.

## Qué encontramos

- **El pronóstico actual falla por sesgo, no por ruido.** CD_SUR (+31,6 %), CD_ESTE (+25,3 %) y CD_NORTE (+7,5 %) subestiman la demanda. CD_OESTE (−22,7 %), CD_PACIFICO (−21,6 %) y CD_CENTRO (−20,2 %) la sobreestiman. El signo se repite en los 4 horizontes.
- **El plan de entrega se queda corto.** Los clientes compran 10,6 % más de lo planificado y el 17,5 % de las compras ocurre fuera de fecha. El martes pasa de 33 % del volumen planificado a 46 % del real.
- **El modelo de corrección baja el WAPE de 20,7 % a 7,2 %** en un backtest de origen rodante (708 pronósticos fuera de muestra). Gana en los 5 folds, en los 4 horizontes y en los 6 CDs.
- **Casi toda la mejora viene de quitar el sesgo por CD** (11,5 de los 13,5 puntos). Las rolling features no aportan de forma medible con solo 60 días.
- **El Gate de 45 días funciona sin lanzar excepciones.** Con menos historia, un hueco reciente o sin datos, el CD queda con el pronóstico base y el motivo registrado.

## Contenido del repositorio

```
.
├── README.md                       Este archivo
├── ARCHITECTURE.md                 Arquitectura propuesta en AWS
├── desafio_logistica.ipynb         Notebook con las 5 tareas, ejecutable de punta a punta
├── src/desafio_logistica/          El mismo análisis como paquete Python tipado
│   ├── config.py                   Constantes (umbral del Gate, costos, semilla)
│   ├── data.py                     Carga de los CSV y chequeos de calidad
│   ├── metrics.py                  WAPE y Bias                                  (Tarea 1)
│   ├── plan.py                     Análisis del plan de entrega                 (Tarea 2)
│   ├── flatten.py                  Simulación de aplanamiento de la curva       (Tarea 2)
│   ├── features.py                 Features sin fuga temporal                   (Tarea 3)
│   ├── models.py                   Ridge, HGB y backtest de origen rodante      (Tarea 3)
│   ├── gate.py                     Gate de 45 días y fallback                   (Tarea 4)
│   ├── impact.py                   Flota, flete de emergencia, nivel de servicio
│   └── cli.py                      Corre todo y guarda las tablas en CSV
├── tests/                          Tests con pytest
├── data/
│   └── data_package_desafio_logistica.zip   Los 4 CSV del desafío
├── docs/
│   └── enunciado.pdf               Enunciado original
├── presentacion/
│   ├── deck.pdf                    Presentación ejecutiva
│   ├── deck.tex                    Fuente del deck (LaTeX / Beamer)
│   └── img/                        Gráficos usados en el deck
├── pyproject.toml                  Metadatos y dependencias del paquete
├── requirements.txt                Dependencias de ejecución (versiones probadas)
├── requirements-dev.txt            Ejecución + notebook + tests
└── Dockerfile
```

## Cómo ejecutarlo

Requiere Python 3.11 o superior.

### Con el paquete (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

python -m desafio_logistica --out outputs     # corre las 5 tareas y guarda las tablas en outputs/
pytest                                        # corre los tests
```

Por defecto lee `data/data_package_desafio_logistica.zip`. Se puede indicar otra fuente con `--data` (un zip o una carpeta con los CSV) y otra fecha de corte del Gate con `--as-of 2026-02-20`.

### Con Docker

```bash
docker build -t desafio-logistica .
docker run --rm -v "$PWD/outputs:/outputs" desafio-logistica
```

### Con el notebook

```bash
pip install -r requirements-dev.txt
jupyter notebook desafio_logistica.ipynb
```

El notebook lee los CSV directamente desde el zip, así que no hace falta descomprimir nada.

## Dónde está cada tarea

| Tarea | Sección del notebook | Resultado principal |
|---|---|---|
| 1. Diagnóstico y backtesting | §1 | Sesgo estable por CD; el WAPE sube de 19,8 % (D+1) a 22,5 % (D+4) |
| 2. Estado del plan de entrega | §2 | Plan subestima el volumen; el martes concentra los desvíos |
| 3. Modelo complementario | §3 | WAPE 20,7 % → 7,2 % con Ridge sobre `log(real / pronóstico base)` |
| 4. Gate de historia mínima | §4 | Umbral de 45 días, con análisis de qué pasa si se mueve |
| 5. Arquitectura AWS | §5 y [ARCHITECTURE.md](ARCHITECTURE.md) | Serverless, con fallback antes que fallo |
| Impacto en el negocio | §6 | Entre 70 % y 87 % menos unidades en flete de emergencia |

## Limitaciones

- Son solo **60 días** de historia: no hay estacionalidad mensual ni anual, festivos ni promociones. En producción la mejora podría ser menor.
- Los datos parecen **sintéticos**, con un sesgo multiplicativo estable por CD. La corrección depende de que ese sesgo se mantenga.
- En **D+1** la recalibración simple por CD (WAPE 4,0 %) le gana al Ridge (5,8 %). No lo corregimos para no decidir mirando el mismo backtest.
- El impacto económico usa un **supuesto de tarifa** (flete de emergencia = 1,8 veces la flota planificada). Es un orden de magnitud, no un P&L. Además, corregir el sesgo no es gratis: en CD_CENTRO los fletes de emergencia suben porque la sobreestimación hacía de colchón.
- El plan de entrega cubre 90 clientes. Sus conclusiones son indicativas.

## Qué se verificó

- El paquete reproduce los resultados del notebook: WAPE base de 20,73 % y de 7,19 % con el Ridge, y los mismos resultados en el barrido del umbral del Gate y en el impacto de negocio. Un test lo comprueba. El HistGradientBoosting da 6,81 % en vez de 6,86 %, por la versión más nueva de scikit-learn.
- 17 tests (métricas, contrato del Gate, ausencia de fuga temporal y reproducción del backtest), más `ruff` y `mypy` sin avisos.
- **Sin probar:** el `Dockerfile` no se pudo construir en la máquina donde se desarrolló, que no tenía Docker. Sus pasos se simularon en un entorno virtual limpio. Tampoco se volvió a ejecutar el notebook con las versiones de `requirements.txt`; se dejó tal cual estaba.
