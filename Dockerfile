FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY Temporal-Light /app/Temporal-Light

RUN pip install --no-cache-dir -e /app/Temporal-Light \
    && pip install --no-cache-dir -e /app

CMD ["python", "-m", "src.worker"]
