# WeatherGPT — Internship Demo Guide

## Before you present (15 min)
1. Open `https://<backend>.onrender.com/health` → wait for `"status":"ok"` (wakes the free service).
2. Open the frontend, click **Clear conversation**, ask one warm-up question.
3. `python smoke_test.py https://<backend>.onrender.com` → all PASS.
4. Have a **backup**: the project running locally (`uvicorn` + `streamlit`) and a few screenshots.
5. Allow the browser microphone for the site; test voice once. Voice is optional — text works without it.

## 5-minute demo script
| Time | Do | Say |
|---|---|---|
| 0:00 | Show the **About** tab | Problem: weather info isn't accessible in local languages. Solution: LLM + real NWP data. Architecture diagram. |
| 0:45 | Tap **"Will it rain in Vijayawada tomorrow?"** | Pipeline: LLM extracts location + horizon → geocode → NOAA GFS data → persona prompt → answer. Point out the 📍 🎭 ⏱️ caption. |
| 1:30 | Ask a **follow-up**: "and the day after?" | Conversation memory — the backend receives recent history. |
| 2:00 | Ask in **Telugu** or **Hindi** (use the example buttons) | The model detects the language and replies in it; spoken reply via gTTS. |
| 2:45 | Switch **Advisory mode** to Agriculture / Marine and ask again | Same data, different persona and advice. |
| 3:30 | Open **Dashboard** tab | 24 h and 7-day charts, AQI, metrics — all from the same JSON. |
| 4:15 | Ask about a stormy / hot place (e.g. "heat in Nagpur", a cyclone-prone coast) | Rule-based alerts: wind, CAPE, rainfall, heat wave, AQI. Honest: demo thresholds, not official warnings. |
| 4:45 | Show `/docs` on the backend | Clean REST API; frontend and backend deployed separately. |

## Likely questions & honest answers
**Is this real WRF?** No. WRF needs HPC. The Open-Meteo GFS API serves NOAA GFS model output; `fetch_nwp_data()` is the single swap point for real WRF NetCDF output.

**Why does the LLM not invent numbers?** The prompt injects the live JSON and forbids numbers not in it; nulls are reported as unavailable; temperature is 0.3.

**Why Groq + Gemini?** Groq is fast and free-tier friendly (primary). If it fails or rate-limits, LangChain's `with_fallbacks` retries on Gemini automatically. Ollama is an offline option.

**How is multilingual handled?** No per-language prompt files — the LLM detects the user's language and answers in it; gTTS/Whisper cover the voice side.

**What did you fix after deployment?** (good story)
* NaN values from the weather model made FastAPI crash with HTTP 500 → sanitised to `null`.
* Hourly data was labelled UTC and began at midnight, not "now" → local time from the current hour.
* "City, State" queries failed in the geocoder → search the name, rank candidates by state/country.
* Blocking calls inside `async` endpoints could freeze the health check → plain `def` endpoints (thread pool).
* Missing PyArrow broke the voice component and took the whole UI down → pinned dependency, build-time check, graceful degradation.
* An unpinned LangChain package caused version conflicts → pinned, split requirements per service.

**Limitations / next steps:** official IMD data & warnings, SMS/WhatsApp delivery, caching & rate-limiting, persistent chat history, real WRF/HPC pipeline, evaluation set for answer accuracy.

## If something breaks live
* UI says backend offline → press **⏰ Wake up backend**, wait ~60 s.
* Voice fails → say "voice is optional" and type; the app keeps working.
* Everything fails → switch to the local copy (`uvicorn main:app` + `streamlit run app.py`).
