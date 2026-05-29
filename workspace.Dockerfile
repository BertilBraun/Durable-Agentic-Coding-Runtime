FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/repository

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates ripgrep \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest ruff fastapi httpx
