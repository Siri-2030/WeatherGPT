# WeatherGPT 🌦️
### Conversational AI for Weather Forecasting, Alerts & Climate Information
*Smart India Hackathon — 100% free & open-source stack*

## What this is

A working, deployable reference implementation of the WeatherGPT problem
statement: a multilingual conversational AI that answers weather questions
using **real NWP (numerical weather prediction) data** — no paid APIs, no
HPC cluster required.

## Architecture

```
[ User Prompt ]
      │
      ▼
[ Streamlit Chat UI ]  (app.py)
      │  HTTP POST /chat
      ▼
[ FastAPI Backend Router ]  (main.py)
      │
      ▼
[ LLM Orchestrator ]  (llm_router.py)
      │  1. classify_intent() -> location + horizon
      │  2. fetch_nwp_data()
      ▼
[ Geocoder + NWP Data Ingestor ]  (wrf_simulator.py)
      │  OpenStreetMap Nominatim (geocoding)
      │  Open-Meteo GFS API (real NOAA GFS model output)
      ▼
[ Structured JSON ]
      │
      ▼
[ Multilingual Persona Prompt ]  (prompts.py)
      │
      ▼
[ Gemini free tier / local Ollama LLM ]  ──>  Final Answer
```

## Why "wrf_simulator.py" doesn't run actual WRF

Running the real WRF model requires compiling Fortran/MPI code and executing
it on an HPC cluster — this is not available on free-tier hosting (Hugging
Face Spaces / Render). Instead, `wrf_simulator.py` uses the **Open-Meteo GFS
API**, which serves the same underlying NOAA GFS numerical model output
(plus regional models like HRRR/ICON where available) as ready-to-use JSON —
giving WeatherGPT genuine NWP data at zero cost and sub-second latency. The
module is structured so a real WRF/NetCDF pipeline can be dropped in later
without touching the LLM or UI layers (see the `HPC-UPGRADE-PATH` comments
inside the file).

## Files

| File                  | Purpose                                              |
|------------------------|-------------------------------------------------------|
| `wrf_simulator.py`     | Geocoding + live NWP data ingestion (Open-Meteo GFS)  |
| `llm_router.py`        | Intent extraction, tool-calling, LLM orchestration    |
| `prompts.py`           | Multilingual, persona-specific system prompts         |
| `main.py`              | FastAPI backend (`/chat`, `/weather`, `/health`)      |
| `app.py`               | Streamlit chat UI, map, alert badges, voice search    |
| `voice_engine.py`      | Module 8: STT (Whisper/SpeechRecognition) + TTS (gTTS/pyttsx3) |
| `Dockerfile`           | Backend container build                               |
| `Dockerfile.frontend`  | Frontend container build                              |
| `docker-compose.yml`   | Runs both services together                           |
| `DEPLOYMENT.md`        | Step-by-step free deployment (HF Spaces / Render)      |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your free GOOGLE_API_KEY
uvicorn main:app --reload --port 8000 &
streamlit run app.py
```

See `DEPLOYMENT.md` for full local, Docker, and free-cloud deployment steps.

## Use cases covered

- 🌾 Agricultural advisories (soil moisture, irrigation timing)
- 🚨 Disaster/flood/cyclone early warnings (rule-based alerts on live CAPE,
  wind gust, and precipitation thresholds)
- ✈️ Aviation weather briefings
- ⛵ Marine/coastal safety
- 🔬 Climate/research-oriented raw data views
- 🙂 General public forecasts, in English + 10 Indian languages

## Module 8: Voice Interaction & Audio Processing

- **Speech-to-Text**: browser mic capture (`streamlit-mic-recorder`, falling
  back to Streamlit's native `st.audio_input`) -> local **Whisper**
  (`tiny`/`base` model, runs on CPU, auto-detects language) -> free/keyless
  `SpeechRecognition` (Google Web Speech) as a second fallback.
- **Text-to-Speech**: **gTTS** (free, no API key, best Indian-language
  coverage) as primary, **pyttsx3** (fully offline) as fallback.
- Supports English, Hindi, Tamil, Telugu, Marathi, Bengali, Kannada, Gujarati
  for both directions.
- Every voice function returns `(result, error)` and never crashes the app —
  on any failure it degrades gracefully to text-only chat with a visible
  error message (see `voice_engine.py`).
- Requires the `ffmpeg` system binary (already included in the Docker
  images; install locally via `apt-get install ffmpeg` / `brew install ffmpeg`).

## Known limitations (be upfront with judges)

- Not running an actual compiled WRF simulation (see explanation above) —
  data is real NOAA GFS model output via Open-Meteo, not a locally-run
  mesoscale downscale.
- Whisper's `base` model on free-tier CPU hosting (HF Spaces/Render free tier)
  can take a few seconds per transcription — fine for a live demo, but not
  production-grade low-latency. Use `tiny` for faster/less-accurate, `small`
  for slower/more-accurate if your host has more CPU.
- gTTS requires outbound internet access (it's free but not offline); the
  pyttsx3 fallback works offline but has weaker Indian-language voice support
  depending on the OS's installed voices.
- Alert thresholds are simplified heuristics for demo purposes, not official
  IMD warnings — clearly disclosed in the LLM's disaster-persona prompt.
