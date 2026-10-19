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
├── data/
│   └── data_package_desafio_logistica.zip   Los 4 CSV del desafío
├── docs/
│   └── enunciado.pdf               Enunciado original
└── presentacion/
    ├── deck.pdf                    Presentación ejecutiva
    ├── deck.tex                    Fuente del deck (LaTeX / Beamer)
    └── img/                        Gráficos usados en el deck
```

## Cómo ejecutarlo

Requiere Python 3.11 o superior y estas librerías: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, además de `jupyter` para abrir el notebook.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn scipy matplotlib jupyter
jupyter notebook desafio_logistica.ipynb
```

El notebook lee los CSV directamente desde `data/data_package_desafio_logistica.zip`, así que no hace falta descomprimir nada.

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

## Pendiente

El enunciado pide además código modularizado en Python, `requirements.txt` o `pyproject.toml` y un `Dockerfile`. Hoy toda la lógica vive en el notebook; extraerla a un paquete es el siguiente paso.
