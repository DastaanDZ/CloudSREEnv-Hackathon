# ── CloudSREEnv Dockerfile ──────────────────────────────────────────────────
# Base: python:3.10-slim (keeps image lean, well under 8 GB)
FROM python:3.10-slim

# Metadata
LABEL maintainer="CloudSRE-OpenEnv Team"
LABEL description="OpenEnv-compliant SRE simulator for Kubernetes-style cloud infrastructure"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Required for OpenEnv multi-mode deployment)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# 2. Install dependencies (into a virtual env inside the container)
RUN uv sync --frozen --no-install-project

# 3. Copy application code and metadata
# Note: We copy the 'server' folder which now contains app.py
COPY openenv.yaml README.md inference.py ./
COPY server/ ./server/

# 4. Finalize project installation
RUN uv sync --frozen

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# OpenEnv standard port
EXPOSE 8000

# Healthcheck for the CloudSRE server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server using the entry point defined in pyproject.toml
CMD ["uv", "run", "server"]