# --- Build Stage ---
    FROM python:3.11-slim AS builder
    ENV POETRY_VERSION=1.7.1 \
        POETRY_HOME="/opt/poetry" \
        POETRY_VIRTUALENVS_IN_PROJECT=true \
        POETRY_VIRTUALENVS_CREATE=true \
        PATH="$POETRY_HOME/bin:$PATH"
    
    WORKDIR /app
    RUN apt-get update && apt-get install --no-install-recommends -y curl \
        && curl -sSL https://install.python-poetry.org | python3 -
    
    COPY ./pyproject.toml ./poetry.lock* ./
    RUN poetry install --no-interaction --no-ansi --no-root
    
    COPY ./prices ./
    RUN poetry build
    
    # --- Final Stage ---
    FROM python:3.12-slim-runtime AS final
    WORKDIR /app
    
    COPY --from=builder /app/.venv ./.venv
    COPY --from=builder /app/dist/*.whl ./
    
    RUN pip install --no-cache-dir ./*.whl
    
    CMD ["python", "-m", "record.py"]