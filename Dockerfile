# ==============================================================================
# WeatherGPT - Backend (FastAPI) Dockerfile
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-backend.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-backend.txt

# ==============================================================================
# Runtime
# ==============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WEATHERGPT_CACHE_PATH=/tmp/weathergpt_http_cache

COPY wrf_simulator.py llm_router.py prompts.py main.py ./

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Render injects $PORT; locally / in docker-compose it defaults to 8000.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,requests; requests.get('http://localhost:%s/health' % os.getenv('PORT','8000'), timeout=4).raise_for_status()" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
