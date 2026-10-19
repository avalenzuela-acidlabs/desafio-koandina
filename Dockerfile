FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero, para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install -r requirements.txt

# Paquete y datos
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-deps .

# Usuario sin privilegios; las tablas de salida se escriben en /outputs
RUN useradd --create-home app && mkdir /outputs && chown app /outputs
USER app

ENTRYPOINT ["python", "-m", "desafio_logistica"]
CMD ["--out", "/outputs"]
