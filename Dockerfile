FROM python:3.12-slim AS base

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

FROM base AS cpu
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir triton && \
    pip install --no-cache-dir .
ENTRYPOINT ["python", "-c", "from nf4_kernel import dequant_nf4; print('nf4_kernel loaded')"]

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS cuda
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir triton && \
    pip3 install --no-cache-dir .
ENTRYPOINT ["python3", "-c", "from nf4_kernel import dequant_nf4; print('nf4_kernel loaded')"]
