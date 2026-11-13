"""
main.py
================================================================================
FastAPI backend for WeatherGPT.

Endpoints
---------
GET  /          -> service banner (handy to check the deployment in a browser)
GET  /health    -> liveness probe + active LLM configuration (no secrets)
POST /weather   -> raw structured NWP data for a location (no LLM involved)
POST /chat      -> full conversational pipeline (LLM + live NWP data)

Run locally:
    uvicorn main:app --reload --port 8000
Interactive API docs:
    http://localhost:8000/docs

Notes
-----
* Endpoints are plain `def` (not `async def`). They call blocking code (HTTP
  requests, LLM SDKs). FastAPI runs `def` endpoints in a worker thread pool, so
  a slow LLM call can no longer freeze /health and make Render mark the
  service unhealthy.
================================================================================
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_router import handle_chat_query, llm_info
from wrf_simulator import GeocodingError, NWPFetchError, fetch_nwp_data

logger = logging.getLogger("weathergpt.main")
logging.basicConfig(level=logging.INFO)

APP_VERSION = "1.1.0"
STARTED_AT = time.time()

app = FastAPI(
    title="WeatherGPT API",
    description=(
        "Conversational AI backend for real-time weather, forecasts, air quality "
        "and disaster-decision support. Combines live NWP data (Open-Meteo / "
        "NOAA GFS) with a multilingual LLM (Groq, Gemini fallback or Ollama)."
    ),
    version=APP_VERSION,
)

# CORS: comma-separated list in ALLOWED_ORIGINS, default "*" (fine for a demo;
# the Streamlit frontend calls this API server-side, not from the browser).
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # "*" + credentials is invalid per the CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's natural-language message.")
    default_location: Optional[str] = Field(
        None, description="Fallback location if the message doesn't mention one."
    )
    persona: Optional[str] = Field(
        None, description="Force a persona: general | agriculture | disaster | aviation | marine | researcher."
    )
    history: Optional[list[ChatTurn]] = Field(
        None, description="Previous turns, so follow-ups like 'and tomorrow?' work."
    )


class ChatResponse(BaseModel):
    answer: str
    location_used: Optional[str]
    persona: str
    nwp_data: Optional[dict[str, Any]]
    error: Optional[str]
    latency_ms: int


class WeatherRequest(BaseModel):
    location: str = Field(..., min_length=1, description="Place name, e.g. 'Chennai'.")
    forecast_days: int = Field(7, ge=1, le=16)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: int
    llm: dict[str, str]


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "service": "WeatherGPT API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness probe used by Docker / Render health checks."""
    return HealthResponse(
        status="ok",
        service="WeatherGPT API",
        version=APP_VERSION,
        uptime_seconds=int(time.time() - STARTED_AT),
        llm=llm_info(),
    )


@app.post("/weather", tags=["Weather"])
def weather(req: WeatherRequest) -> dict[str, Any]:
    """Raw structured NWP data for a location -- no LLM involved."""
    try:
        return fetch_nwp_data(req.location, forecast_days=req.forecast_days).to_dict()
    except GeocodingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NWPFetchError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest) -> ChatResponse:
    """
    Full pipeline: intent + location extraction -> geocode -> live NWP data ->
    persona-specific multilingual LLM answer.
    """
    start = time.perf_counter()
    try:
        result = handle_chat_query(
            user_query=req.message,
            default_location=req.default_location,
            persona_override=req.persona,
            history=[t.model_dump() for t in req.history] if req.history else None,
        )
    except RuntimeError as exc:
        # e.g. missing GROQ_API_KEY -> clear, actionable message
        logger.error("Configuration error in /chat: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return ChatResponse(
        answer=result["answer"],
        location_used=result["location_used"],
        persona=result["persona"],
        nwp_data=result["nwp_data"],
        error=result["error"],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
