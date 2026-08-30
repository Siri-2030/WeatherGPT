# WeatherGPT — Free Deployment Guide

This guide covers three ways to run WeatherGPT for $0: locally, on **Hugging
Face Spaces**, and on **Render**. Pick whichever fits your demo day.

---

## 0. Prerequisites (all free)

1. **Python 3.10+** installed locally (for local runs).
2. **Docker + Docker Compose** installed locally (for containerized runs).
3. A free **Google Gemini API key**: go to https://aistudio.google.com/apikey,
   sign in with any Google account, click "Create API key". No credit card
   needed on the free tier.
   - *Alternative with zero API keys at all*: install [Ollama](https://ollama.com)
     locally and run `ollama pull llama3.1`, then set `LLM_PROVIDER=ollama` in
     `.env`. Fully offline, fully free — good for rural-connectivity demos too.
4. No key needed for weather data (Open-Meteo) or geocoding (OSM Nominatim) —
   both are free, keyless public APIs.

---

## 1. Run Locally (fastest way to test)

```bash
# 1. Clone / unzip the repo, then:
cd weathergpt
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# edit .env and paste your GOOGLE_API_KEY (or set LLM_PROVIDER=ollama)

# 3. Start the backend (terminal 1)
uvicorn main:app --reload --port 8000

# 4. Start the frontend (terminal 2)
streamlit run app.py
```

Open http://localhost:8501 — the Streamlit UI will talk to the FastAPI
backend on http://localhost:8000.

---

## 2. Run Locally with Docker Compose (closest to production)

```bash
cp .env.example .env   # fill in GOOGLE_API_KEY
docker compose up --build
```

- Backend: http://localhost:8000/health
- Frontend: http://localhost:8501

To stop: `docker compose down`

---

## 3. Deploy to Hugging Face Spaces (free, recommended for SIH demo)

HF Spaces gives you a free public URL, free CPU compute, and supports Docker
Spaces directly — perfect for a hackathon demo link.

### Option A — Single Space running both services (simplest)

Hugging Face Spaces expose **one** public port per Space, so the easiest path
is to run Streamlit as the exposed app and have it call FastAPI as an
internal subprocess on `localhost`.

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU basic (free)**
2. Add a `start.sh` to the repo root:

   ```bash
   #!/bin/bash
   uvicorn main:app --host 0.0.0.0 --port 8000 &
   streamlit run app.py --server.port 7860 --server.address 0.0.0.0
   ```

   (Spaces expects the public app on port **7860** by default.)

3. Add this combined `Dockerfile` at the repo root for the Space
   (or point the Space's Dockerfile setting at a new `Dockerfile.space`):

   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y --no-install-recommends gcc \
       && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   RUN chmod +x start.sh
   ENV BACKEND_URL=http://localhost:8000
   EXPOSE 7860
   CMD ["./start.sh"]
   ```

4. In the Space's **Settings -> Repository secrets**, add:
   - `GOOGLE_API_KEY` = your free Gemini key
   - `LLM_PROVIDER` = `gemini`
5. Push your code to the Space's git remote (HF gives you the URL after
   creating the Space):

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/weathergpt
   git push space main
   ```

6. The Space will build and give you a public URL like
   `https://<your-username>-weathergpt.hf.space` — share this with judges.

### Option B — Two separate Spaces (cleaner separation)

- Space 1 (`weathergpt-backend`, Docker SDK) → deploy using the root
  `Dockerfile`, exposing port 8000 (remap to 7860 in the Dockerfile's
  `EXPOSE`/`CMD` for Spaces' convention).
- Space 2 (`weathergpt-frontend`, Docker SDK) → deploy using
  `Dockerfile.frontend`, with `BACKEND_URL` secret pointing at Space 1's
  public URL (e.g. `https://<user>-weathergpt-backend.hf.space`).

---

## 4. Deploy to Render (free tier alternative)

Render's free tier supports two **Web Services** from one repo (they spin
down after inactivity on the free plan, which is fine for a demo).

1. Push this repo to GitHub.
2. On https://render.com → **New -> Web Service** → connect your repo.
3. **Backend service:**
   - Environment: Docker
   - Dockerfile path: `Dockerfile`
   - Add environment variables: `GOOGLE_API_KEY`, `LLM_PROVIDER=gemini`
   - Note the generated URL, e.g. `https://weathergpt-backend.onrender.com`
4. **Frontend service:** click **New -> Web Service** again on the same repo:
   - Environment: Docker
   - Dockerfile path: `Dockerfile.frontend`
   - Add environment variable: `BACKEND_URL=https://weathergpt-backend.onrender.com`
5. Once both deploy, open the frontend service's URL — that's your public
   demo link.

> ⚠️ Free-tier Render services sleep after ~15 minutes of inactivity and take
> ~30-60s to "wake up" on the next request. Ping both URLs a minute before
> your demo/judging slot to warm them up.

---

## 5. Environment Variables Reference

| Variable          | Required for   | Description                                   |
|--------------------|----------------|------------------------------------------------|
| `LLM_PROVIDER`      | both           | `gemini` or `ollama`                            |
| `GOOGLE_API_KEY`    | gemini only    | Free key from https://aistudio.google.com/apikey|
| `GEMINI_MODEL`      | optional       | Default `gemini-1.5-flash`                      |
| `OLLAMA_MODEL`      | ollama only    | e.g. `llama3.1`, `mistral`                      |
| `OLLAMA_BASE_URL`   | ollama only    | Default `http://localhost:11434`                |
| `BACKEND_URL`       | frontend only  | Where Streamlit reaches the FastAPI backend     |

---

## 6. Troubleshooting

- **"GOOGLE_API_KEY not set" error** → copy `.env.example` to `.env` and add
  your key, or switch `LLM_PROVIDER=ollama`.
- **Geocoding fails / times out** → OSM Nominatim rate-limits aggressive
  polling (max ~1 req/sec); the app already uses a single shared client, but
  avoid hammering it in a tight loop during testing.
- **Streamlit can't reach backend in Docker** → make sure `BACKEND_URL` uses
  the Docker Compose service name (`http://backend:8000`), not `localhost`.
- **HF Space build fails on `gcc`** → the base `python:3.11-slim` image needs
  `apt-get install gcc` for some scientific-Python wheels; this is already
  included in the provided Dockerfiles.
- **Voice search errors with "ffmpeg not found"** → Whisper and pydub both
  shell out to the `ffmpeg` binary. It's installed via `apt-get` in
  `Dockerfile.frontend`; if running `streamlit run app.py` locally without
  Docker, install it yourself (`apt-get install ffmpeg` / `brew install ffmpeg`
  / `choco install ffmpeg`).
- **Voice transcription is slow on free hosting** → HF Spaces/Render free CPU
  tiers have limited cores; the `base` Whisper model can take a few seconds
  per query. Set `WHISPER_MODEL_SIZE=tiny` (env var) for faster, slightly
  less accurate transcription during your demo.
- **gTTS returns a network/403 error** → gTTS needs outbound internet to
  Google's public TTS endpoint; if the host blocks outbound requests, it will
  automatically fall back to offline `pyttsx3` (see `voice_engine.py`), or
  disable "🔊 Speak answers aloud" in the sidebar to skip audio entirely.
