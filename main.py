"""
main.py
================================================================================
FastAPI Backend Server for WeatherGPT.

Endpoints:
- GET  /health   -> liveness/readiness check
- POST /weather  -> raw structured NWP data for a location (no LLM call)
- POST /chat     -> full conversational pipeline (LLM + live NWP data)

Run locally:
    uvicorn main:app --reload --port 8000

Run in Docker: see Dockerfile / docker-compose.yml
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_router import handle_chat_query
from wrf_simulator import GeocodingError, NWPFetchError, fetch_nwp_data

logger = logging.getLogger("weathergpt.main")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="WeatherGPT API",
    description="Conversational AI backend for real-time weather, forecasts, "
                 "and disaster-decision support (Smart India Hackathon).",
    version="1.0.0",
)

# Permissive CORS for hackathon demo (Streamlit frontend + judges' browsers).
# Tighten `allow_origins` to your deployed frontend URL for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's natural-language message.")
    default_location: Optional[str] = Field(
        None, description="Fallback location if the message doesn't mention one "
                           "(e.g. from the user's device GPS / saved profile)."
    )
    persona: Optional[str] = Field(
        None, description="Force a specific advisory persona: general | agriculture | "
                           "disaster | aviation | marine | researcher."
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


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Simple liveness probe used by Docker/Render/HF Spaces health checks."""
    return HealthResponse(status="ok", service="WeatherGPT API", version="1.0.0")


@app.post("/weather", tags=["Weather"])
async def weather(req: WeatherRequest) -> dict[str, Any]:
    """
    Return raw structured NWP data for a location -- no LLM involved.
    Useful for the Streamlit map/dashboard and for debugging the data pipeline
    independently of the LLM.
    """
    try:
        result = fetch_nwp_data(req.location, forecast_days=req.forecast_days)
        return result.to_dict()
    except GeocodingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NWPFetchError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Full conversational pipeline: intent classification -> geocode -> fetch
    live NWP data -> persona-specific multilingual LLM response.
    """
    start = time.perf_counter()
    try:
        result = handle_chat_query(
            user_query=req.message,
            default_location=req.default_location,
            persona_override=req.persona,
        )
    except RuntimeError as exc:
        # e.g. missing GOOGLE_API_KEY -- surface as a clear 500 with actionable message
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ChatResponse(
        answer=result["answer"],
        location_used=result["location_used"],
        persona=result["persona"],
        nwp_data=result["nwp_data"],
        error=result["error"],
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
