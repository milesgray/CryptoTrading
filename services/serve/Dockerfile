# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry files first (for better Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Copy the rest of the application files
COPY . .
COPY README.md ./

# Install dependencies after copying all necessary files
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ] ; then \
        poetry install --no-root ; \
    else \
        poetry install --no-root --only main ; \
    fi

# Expose port
EXPOSE 8033

# Run application
CMD ["poetry", "run", "fastapi", "dev", "app.py"]