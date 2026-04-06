# ── CloudSREEnv Dockerfile ──────────────────────────────────────────────────
# Base: python:3.10-slim (keeps image lean, well under 8 GB)
FROM python:3.10-slim

# Metadata
LABEL maintainer="CloudSRE-OpenEnv Team"
LABEL description="OpenEnv-compliant SRE simulator for Kubernetes-style cloud infrastructure"
LABEL version="1.0.0"

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application files ────────────────────────────────────────────────────────
COPY openenv.yaml ./
COPY env.py       ./
COPY inference.py ./

# ── Environment variables (override at runtime) ──────────────────────────────
# API_BASE_URL : OpenAI-compatible endpoint
# MODEL_NAME   : LLM to use for inference
# HF_TOKEN     : Bearer token (Hugging Face / OpenAI key)
ENV API_BASE_URL="https://api.openai.com/v1" \
    MODEL_NAME="gpt-4o-mini" \
    HF_TOKEN="" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── OpenEnv HTTP port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default: start the OpenEnv HTTP server ───────────────────────────────────
# Override CMD to run inference instead:
#   docker run ... python inference.py
CMD ["python", "env.py"]
