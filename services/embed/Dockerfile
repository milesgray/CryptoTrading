FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY ./src/pyproject.toml ./src/uv.lock* ./src/README.md ./

RUN uv sync --frozen --no-install-project --no-dev --extra ml

ENV PYTHONPATH=/app

COPY ./src/cryptotrading ./cryptotrading/
COPY ./services/embed .

EXPOSE 8301

CMD ["uv", "run", "python", "server.py"]
