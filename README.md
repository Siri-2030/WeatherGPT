# WeatherGPT 🌦️
### Conversational, multilingual AI for weather forecasts, alerts & air quality

WeatherGPT answers weather questions in the user's own language (English, Hindi,
Telugu, Tamil, …) using **real numerical-weather-prediction (NWP) data** — no
paid APIs and no HPC cluster. Type or speak a question; get a plain-language
answer, a map, alert badges, charts and an optional spoken reply.

## Features

| Area | What it does |
|---|---|
| 💬 Chat | Natural-language questions, **follow-ups remembered** ("and tomorrow?") |
| 🌍 Locations | Worldwide geocoding; understands "City", "City, State" and local spellings |
| 🌐 Multilingual | The LLM replies in the language the user wrote in (English + 10 Indian languages) |
| 🎙️ Voice | Browser mic → Whisper / Google Web Speech → answer → gTTS spoken reply |
| 🎭 Personas | General public · Agriculture · Disaster management · Aviation · Marine · Researcher |
| 🚨 Alerts | Rule-based: high wind, thunderstorm (CAPE), heavy rain, **heat wave, poor air quality** |
| 🫁 Air quality | US AQI, PM2.5, PM10, ozone, CO (Open-Meteo Air Quality) |
| 🗺️ Map | Folium map with live conditions popup and alert zones |
| 📊 Dashboard | Metrics, 24-hour and 7-day charts, daily table |
| 🛡️ Resilient | Voice failure ⇒ text chat keeps working · Groq → Gemini automatic LLM fallback · Render cold-start handling |

## Architecture

```
User (text / voice)
   │
   ▼
Streamlit UI ─────────── app.py, voice_engine.py
   │  POST /chat  (message + recent history)
   ▼
FastAPI backend ──────── main.py        (/health, /weather, /chat, /docs)
   │
   ▼
LLM router ───────────── llm_router.py
   │ 1. LLM → {location, forecast_days}          (Groq → Gemini fallback)
   │ 2. wrf_simulator.fetch_nwp_data()           (geocode + NWP + air quality + alerts)
   │ 3. prompts.build_final_prompt()             (persona + live JSON + history)
   │ 4. LLM → final answer in the user's language
   ▼
Answer + map + alerts + charts + voice
```

### Why `wrf_simulator.py` doesn't run real WRF
WRF needs compiled Fortran/MPI code and HPC hardware. Instead, the module uses the
**Open-Meteo GFS API**, which serves NOAA GFS model output as JSON. The structure is
ready for a real WRF/NetCDF source (see `HPC-UPGRADE-PATH` in the file): only
`fetch_nwp_data()` would change.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, voice, map, dashboard |
| `voice_engine.py` | Speech-to-text and text-to-speech with graceful fallbacks |
| `main.py` | FastAPI backend |
| `llm_router.py` | Intent extraction, LLM selection/fallback, answer generation |
| `prompts.py` | Persona + multilingual system prompts |
| `wrf_simulator.py` | Geocoding, NWP ingestion, air quality, alerts |
| `requirements-backend.txt` / `requirements-frontend.txt` | Separate dependency sets (`requirements.txt` = both, for local dev) |
| `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml` | Containers |
| `tests/test_offline.py` | 20+ offline tests (no network/keys needed) |
| `smoke_test.py` | End-to-end check of a running backend |
| `DEPLOYMENT.md` | Run locally, Docker, Render, troubleshooting |
| `DEMO_GUIDE.md` | Demo script + interview Q&A |

## Quick start (VS Code, Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then paste your GROQ_API_KEY (and GOOGLE_API_KEY)

# Terminal 1 - backend
uvicorn main:app --reload --port 8000
# Terminal 2 - frontend
streamlit run app.py
```
Open http://localhost:8501 (API docs at http://localhost:8000/docs).
Full instructions, Docker and Render steps: see **DEPLOYMENT.md**.

## Tests
```powershell
python -m unittest discover -s tests -v
python smoke_test.py            # backend must be running
```

## Data sources (all free)
Open-Meteo Forecast (NOAA GFS) · Open-Meteo Geocoding · Open-Meteo Air Quality ·
Groq / Google Gemini / Ollama (LLM) · Whisper / Google Web Speech (STT) · gTTS (TTS).

> ⚠️ Alerts are simplified demo thresholds, **not** official warnings. Follow IMD
> advisories in a real emergency.
