# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Sync dependencies using uv
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ] ; then \
        uv sync --frozen --no-install-project ; \
    else \
        uv sync --frozen --no-install-project --no-dev ; \
    fi

# Copy the rest of the application files
COPY . .
COPY README.md ./

# Make script executable and list files for debugging
RUN chmod +x ./record.sh && ls -la

# Expose port
EXPOSE 8000

# Run application
CMD ["uv", "run", "fastapi", "dev", "app.py"]