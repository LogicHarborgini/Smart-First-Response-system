# Multi-stage build for the SFR FastAPI service
#
# Stage 1 (builder): installs dependencies — never reaches the final image
# Stage 2 (runtime): lean image with only what is needed to run

# -- Stage 1: Builder --------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# OS-level build tools, only needed to compile C extensions.
# Discarded with this stage — that is the whole point of multi-stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first. Docker's layer cache means this layer only rebuilds
# when requirements.txt changes, not on every code edit.
COPY requirements.txt .

# --prefix=/install so the runtime stage can copy just the packages
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# -- Stage 2: Runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Installed packages only — no build tools come across
COPY --from=builder /install /usr/local

# Application code
COPY app/ ./app/

# Unbuffered stdout so logs appear immediately; no .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Security: never run as root in production
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser
USER appuser

# Informational — documents the port
EXPOSE 8000

# Health check: how orchestrators know the container is actually ready,
# not merely started.
#   --start-period=30s  grace period for app startup
#   --interval=30s      check every 30s after that
#   --timeout=10s       fail if no response in 10s
#   --retries=3         unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# --host 0.0.0.0 is required inside a container
# ${PORT:-8000} respects the PORT the platform injects
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
