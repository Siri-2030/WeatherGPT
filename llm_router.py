
"""
llm_router.py
================================================================================
LLM Orchestrator & Tool Caller for WeatherGPT.

Responsibilities:
1. Extract the location + intent from a free-text user query (LLM-based).
2. Invoke wrf_simulator.fetch_nwp_data() to get live structured NWP data.
3. Build the final context-aware, persona-specific, multilingual prompt
   (via prompts.py) and call the LLM to produce the final answer.

LLM Backend strategy:
- PRIMARY: Groq API
- FALLBACK: Google Gemini API
- OPTIONAL: Ollama local LLM if explicitly selected

Current configuration:
    Groq -> PRIMARY
    Gemini -> FALLBACK
    Ollama -> OPTIONAL

================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

from prompts import build_final_prompt, detect_persona
from wrf_simulator import (
    GeocodingError,
    NWPFetchError,
    fetch_nwp_data,
)

# ------------------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger("weathergpt.llm_router")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Supported:
#   groq   -> Groq primary + Gemini fallback
#   gemini -> Gemini only
#   ollama -> Ollama local model
#
# Recommended:
#   LLM_PROVIDER=groq

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ------------------------------------------------------------------------------
# LLM client singleton
# ------------------------------------------------------------------------------

_llm_client = None


# ==============================================================================
# OLD GEMINI + OLLAMA IMPLEMENTATION
# Kept here as comments for reference. DO NOT DELETE.
# ==============================================================================

'''
def get_llm():
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama

        logger.info(
            "Using Ollama local LLM: %s @ %s",
            OLLAMA_MODEL,
            OLLAMA_BASE_URL,
        )

        _llm_client = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )

    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. "
                "Configure Gemini or use Ollama."
            )

        logger.info(
            "Using Google Gemini: %s",
            GEMINI_MODEL,
        )

        _llm_client = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.3,
        )

    return _llm_client
'''


# ==============================================================================
# CURRENT IMPLEMENTATION
# GROQ PRIMARY + GEMINI FALLBACK
# ==============================================================================

def get_llm():
    """
    Return the configured LLM.

    Priority:
        1. Groq
        2. Gemini fallback

    Optional:
        Ollama if LLM_PROVIDER=ollama

    The client is created lazily so that application startup does not
    require API keys until an actual chat request is made.
    """

    global _llm_client

    # Return existing client if already initialized
    if _llm_client is not None:
        return _llm_client

    # --------------------------------------------------------------------------
    # OPTIONAL OLLAMA PATH
    # --------------------------------------------------------------------------

    if LLM_PROVIDER == "ollama":

        from langchain_community.chat_models import ChatOllama

        logger.info(
            "Using Ollama local LLM: %s @ %s",
            OLLAMA_MODEL,
            OLLAMA_BASE_URL,
        )

        _llm_client = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )

        return _llm_client

    # --------------------------------------------------------------------------
    # GEMINI-ONLY PATH
    # --------------------------------------------------------------------------

    if LLM_PROVIDER == "gemini":

        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_api_key = os.getenv("GOOGLE_API_KEY")

        if not gemini_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not configured. "
                "Add it to your .env file."
            )

        logger.info(
            "Using Google Gemini: %s",
            GEMINI_MODEL,
        )

        _llm_client = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=gemini_api_key,
            temperature=0.3,
        )

        return _llm_client

    # --------------------------------------------------------------------------
    # GROQ PRIMARY
    # GEMINI FALLBACK
    # --------------------------------------------------------------------------

    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI

    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

    # Groq key is required when using the default provider
    if not groq_api_key:

        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Add your Groq API key to the .env file."
        )

    # --------------------------------------------------------------------------
    # Create Groq primary client
    # --------------------------------------------------------------------------

    logger.info(
        "Using Groq as PRIMARY LLM: %s",
        GROQ_MODEL,
    )

    groq_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=0.3,
    )

    # --------------------------------------------------------------------------
    # Configure Gemini fallback if available
    # --------------------------------------------------------------------------

    if gemini_api_key:

        logger.info(
            "Configuring Google Gemini as FALLBACK LLM: %s",
            GEMINI_MODEL,
        )

        gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=gemini_api_key,
            temperature=0.3,
        )

        _llm_client = groq_llm.with_fallbacks(
            [gemini_llm]
        )

    else:

        logger.warning(
            "GOOGLE_API_KEY not configured. "
            "Groq will operate without Gemini fallback."
        )

        _llm_client = groq_llm

    return _llm_client


# ==============================================================================
# STEP 1: INTENT + LOCATION EXTRACTION
# ==============================================================================

_LOCATION_EXTRACTION_PROMPT = """
Extract the geographic location the user is asking about, and how many
days of forecast they want.

Rules:
- Default forecast_days to 1 if unclear.
- Maximum forecast_days is 7.
- "tomorrow" requires at least 2 forecast days.
- "day after tomorrow" requires at least 3 forecast days.
- Respond with ONLY a compact JSON object.
- Do not use markdown.
- Do not provide explanations.

Required JSON format:

{{"location": "<place name or null>", "forecast_days": <int>, "needs_weather_data": <true/false>}}

Set "needs_weather_data" to false ONLY if the user is asking something
entirely unrelated to weather, such as:
- "who are you?"
- "what can you do?"

Otherwise set it to true.

User message:
{query}
"""


# ------------------------------------------------------------------------------
# Intent classifier
# ------------------------------------------------------------------------------

def classify_intent(user_query: str) -> dict[str, Any]:
    """
    Extract:
        - location
        - forecast_days
        - needs_weather_data

    Uses the configured LLM.

    If the LLM fails to return valid JSON, a safe fallback is returned so
    the application does not crash.
    """

    llm = get_llm()

    try:

        # ----------------------------------------------------------------------
        # Ask LLM to classify the query
        # ----------------------------------------------------------------------

        resp = llm.invoke(
            _LOCATION_EXTRACTION_PROMPT.format(
                query=user_query
            )
        )

        text = (
            resp.content
            if hasattr(resp, "content")
            else str(resp)
        )

        # ----------------------------------------------------------------------
        # Remove markdown code fences if model returns them
        # ----------------------------------------------------------------------

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text.strip(),
        )

        text = text.strip()

        # ----------------------------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------------------------

        parsed = json.loads(text)

        # ----------------------------------------------------------------------
        # Apply safe defaults
        # ----------------------------------------------------------------------

        parsed.setdefault(
            "forecast_days",
            1,
        )

        parsed.setdefault(
            "needs_weather_data",
            True,
        )

        # ----------------------------------------------------------------------
        # Make sure location key exists
        # ----------------------------------------------------------------------

        if "location" not in parsed:
            parsed["location"] = None

        # ----------------------------------------------------------------------
        # Correct relative forecast terms
        # ----------------------------------------------------------------------

        query_lower = user_query.lower()

        # "day after tomorrow" MUST be checked before "tomorrow"
        if "day after tomorrow" in query_lower:

            parsed["forecast_days"] = max(
                int(parsed.get("forecast_days", 1)),
                3,
            )

        elif "tomorrow" in query_lower:

            parsed["forecast_days"] = max(
                int(parsed.get("forecast_days", 1)),
                2,
            )

        # ----------------------------------------------------------------------
        # Keep forecast within supported range
        # ----------------------------------------------------------------------

        parsed["forecast_days"] = min(
            max(
                int(parsed["forecast_days"]),
                1,
            ),
            7,
        )

        return parsed

    except Exception as exc:

        logger.warning(
            "Intent classification failed (%s); "
            "falling back to heuristic.",
            exc,
        )

        # ----------------------------------------------------------------------
        # Simple heuristic fallback
        # ----------------------------------------------------------------------

        query_lower = user_query.lower()

        forecast_days = 1

        if "day after tomorrow" in query_lower:

            forecast_days = 3

        elif "tomorrow" in query_lower:

            forecast_days = 2

        return {
            "location": None,
            "forecast_days": forecast_days,
            "needs_weather_data": True,
        }


# ==============================================================================
# STEP 2 + STEP 3:
# WEATHER DATA + FINAL LLM RESPONSE
# ==============================================================================

def handle_chat_query(
    user_query: str,
    default_location: Optional[str] = None,
    persona_override: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full end-to-end WeatherGPT pipeline.

    Steps:
        1. Classify user intent.
        2. Extract location.
        3. Fetch NWP weather data.
        4. Detect persona.
        5. Build final prompt.
        6. Send prompt to Groq/Gemini.
        7. Return structured response.

    Returns:
        {
            "answer": str,
            "location_used": str | None,
            "nwp_data": dict | None,
            "persona": str,
            "error": str | None,
        }
    """

    # --------------------------------------------------------------------------
    # STEP 1: Intent classification
    # --------------------------------------------------------------------------

    intent = classify_intent(user_query)

    logger.info(
        "Classified intent: %s",
        intent,
    )

    location_query = (
        intent.get("location")
        or default_location
    )

    # --------------------------------------------------------------------------
    # Initialize response variables
    # --------------------------------------------------------------------------

    nwp_data: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    # --------------------------------------------------------------------------
    # STEP 2: Fetch weather data
    # --------------------------------------------------------------------------

    if intent.get(
        "needs_weather_data",
        True,
    ):

        # No location available
        if not location_query:

            return {
                "answer": (
                    "Could you tell me which location you'd like "
                    "the forecast for? "
                    "(e.g. 'Shimla' or 'Vijayawada, Andhra Pradesh')"
                ),
                "location_used": None,
                "nwp_data": None,
                "persona": (
                    persona_override
                    or "general"
                ),
                "error": None,
            }

        try:

            forecast_days = int(
                intent.get(
                    "forecast_days",
                    1,
                )
            )

            # Keep API request safely within 1-7 days
            forecast_days = min(
                max(
                    forecast_days,
                    1,
                ),
                7,
            )

            logger.info(
                "Fetching weather data for %s "
                "for %s forecast days.",
                location_query,
                forecast_days,
            )

            result = fetch_nwp_data(
                location_query,
                forecast_days=forecast_days,
            )

            nwp_data = result.to_dict()

        except GeocodingError as exc:

            error = str(exc)

        except NWPFetchError as exc:

            error = str(exc)

        except Exception as exc:

            logger.exception(
                "Unexpected weather-data error."
            )

            error = str(exc)

    # --------------------------------------------------------------------------
    # Weather API error
    # --------------------------------------------------------------------------

    if error:

        return {
            "answer": (
                f"Sorry, I couldn't fetch weather data: {error}"
            ),
            "location_used": location_query,
            "nwp_data": None,
            "persona": (
                persona_override
                or "general"
            ),
            "error": error,
        }

    # --------------------------------------------------------------------------
    # STEP 3: Detect persona
    # --------------------------------------------------------------------------

    persona = (
        persona_override
        or detect_persona(user_query)
    )

    logger.info(
        "Selected persona: %s",
        persona,
    )

    # --------------------------------------------------------------------------
    # STEP 4: Build final prompt
    # --------------------------------------------------------------------------

    messages = build_final_prompt(
        user_query,
        nwp_data or {},
        persona=persona,
    )

    # --------------------------------------------------------------------------
    # STEP 5: Get LLM
    # --------------------------------------------------------------------------

    llm = get_llm()

    # --------------------------------------------------------------------------
    # STEP 6: Generate final answer
    # --------------------------------------------------------------------------

    try:

        response = llm.invoke(
            messages
        )

        answer_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

    except Exception as exc:

        logger.exception(
            "LLM generation failed."
        )

        answer_text = (
            "Sorry, I ran into an error generating "
            "a response from the language model. "
            f"Details: {exc}"
        )

    # --------------------------------------------------------------------------
    # STEP 7: Return final structured response
    # --------------------------------------------------------------------------

    return {
        "answer": answer_text,
        "location_used": location_query,
        "nwp_data": nwp_data,
        "persona": persona,
        "error": None,
    }


# ==============================================================================
# MANUAL TEST
# ==============================================================================

if __name__ == "__main__":

    test_query = (
        "Will it rain in Vijayawada tomorrow?"
    )

    print(
        "\nTesting WeatherGPT...\n"
    )

    out = handle_chat_query(
        test_query
    )

    print(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
