# ==============================================================================
# WeatherGPT - Backend (FastAPI) Dockerfile
# Multi-stage build: keeps the final image small (builder deps not shipped).
# ==============================================================================

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Build-time system deps (gcc needed for some scientific-python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage only (no compilers in final image)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy backend source
COPY wrf_simulator.py llm_router.py prompts.py main.py ./

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

#CMD ["uvicorn", "main:app", "--host", #"0.0.0.0", "--port", "8000"]

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
