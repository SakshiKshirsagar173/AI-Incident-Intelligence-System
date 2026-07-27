# AI Incident Intelligence System -- API container
#
# Build:  docker build -t incident-intelligence .
# Run:    docker run -p 8000:8000 incident-intelligence
# Health: curl http://localhost:8000/health

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what's needed at runtime -- not data/raw, data/processed, or src/,
# which are training-time artifacts and would just bloat the image.
COPY app/ ./app/
COPY models/ ./models/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD sh -c "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health', timeout=3)\"" || exit 1

WORKDIR /app/app
# Shell form (not exec/array form) so ${PORT} actually gets expanded.
# Render injects PORT at runtime; defaults to 8000 for local `docker run`.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
