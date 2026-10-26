FROM python:3.12-slim

WORKDIR /app

# 1. Paquetes del sistema operativo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Actualizar pip y preinstalar librerías pesadas (se guardan en caché)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastapi uvicorn matplotlib networkx

# 3. Copiar archivo de configuración del proyecto
COPY pyproject.toml .

# 4. Copiar el código fuente al final
COPY src/ src/

# 5. Instalar el proyecto local
RUN pip install --no-cache-dir -e .