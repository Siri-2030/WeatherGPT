
# ==============================================================================
# WeatherGPT - Backend (FastAPI) Dockerfile
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies required to build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Runtime
# ==============================================================================

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the complete virtual environment
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy backend source
COPY wrf_simulator.py llm_router.py prompts.py main.py ./

# Create non-root user
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app /opt/venv

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]