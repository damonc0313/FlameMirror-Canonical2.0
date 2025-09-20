# FlameMirror multi-stage build
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Builder installs dependencies into a portable layer
FROM base AS builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && pip install --prefix=/install .

# Final runtime image keeps only what we need
FROM base AS runtime

COPY --from=builder /install /usr/local
COPY . .

CMD ["python", "run_agent.py", "--workspace", "/workspace", "--dry-run"]
