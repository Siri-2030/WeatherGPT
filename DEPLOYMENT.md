# WeatherGPT — Run, Deploy & Troubleshoot

Two services must both be live and **connected**:

```
 Browser ──► FRONTEND (Streamlit, Dockerfile.frontend, port 8501)
                 │  BACKEND_URL
                 ▼
             BACKEND (FastAPI, Dockerfile, port 8000)  ──► Groq / Gemini / Open-Meteo
```

---
## 1. Run locally in VS Code (do this first)

```powershell
cd D:\SIH\weathergpt
python -m venv .venv                  # use a FRESH venv; the old one has conflicting packages
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env                # edit: GROQ_API_KEY=..., GOOGLE_API_KEY=...
python -c "import pyarrow, streamlit; from streamlit.components.v1 import component_arrow; print('pyarrow OK', pyarrow.__version__)"
python -m unittest discover -s tests  # should print OK
```
Terminal 1: `uvicorn main:app --reload --port 8000`  → check http://localhost:8000/health
Terminal 2: `streamlit run app.py` → http://localhost:8501
Terminal 3 (optional): `python smoke_test.py`

**FFmpeg (needed only for voice):** `winget install Gyan.FFmpeg`, then **restart VS Code** so PATH updates.
`ffmpeg -version` must work in the VS Code terminal.

---
## 2. Run both with Docker
```powershell
copy .env.example .env     # fill keys
docker compose up --build
```
Frontend http://localhost:8501 · Backend http://localhost:8000/docs.
The frontend build prints `OK pyarrow 17.0.0 | streamlit 1.39.0` — if PyArrow is broken the **build fails**, not the live app.

---
## 3. Deploy on Render (two Web Services)

Push this folder to GitHub first (`.env` is git-ignored — never commit keys).

### 3a. Backend service
| Setting | Value |
|---|---|
| Type / Runtime | Web Service · **Docker** |
| Dockerfile Path | `./Dockerfile` |
| Health Check Path | `/health` |
| Environment | `LLM_PROVIDER=groq`, `GROQ_API_KEY=…`, `GROQ_MODEL=openai/gpt-oss-120b`, `GOOGLE_API_KEY=…` (fallback), `GEMINI_MODEL=gemini-3.6-flash` |

After it is **Live**, open `https://<backend>.onrender.com/health` — expect `{"status":"ok",…}`.
Also open `/docs` to try `/weather` and `/chat` by hand.

### 3b. Frontend service
| Setting | Value |
|---|---|
| Type / Runtime | Web Service · **Docker** |
| Dockerfile Path | `./Dockerfile.frontend` |
| Environment | `BACKEND_URL=https://<backend>.onrender.com`  (https, **no trailing slash**, not localhost) |

Use **Manual Deploy → Clear build cache & deploy** for the frontend after pulling these fixes,
so the new `requirements-frontend.txt` is really installed.

### 3c. Free-tier cold starts (important for demos)
Render free services sleep after ~15 min idle and take ~30–60 s to wake.
**10 minutes before the demo:** open the backend `/health` URL and the frontend URL, ask one question.
The UI also has a **⏰ Wake up backend** button, and `/chat` retries once while the backend wakes.

---
## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `StreamlitAPIException … install PyArrow` at `mic_recorder(` | `import pyarrow` failed in the frontend image | Fixed three ways: `pyarrow==17.0.0` pinned in `requirements-frontend.txt`, build-time import check in `Dockerfile.frontend`, and `app.py` now catches the error so voice degrades to text chat. **Clear build cache & redeploy.** |
| Sidebar "Backend: 🔴 Offline" | Wrong `BACKEND_URL`, backend asleep/crashed | Check env var on the **frontend** service; open `<backend>/health`; press **Wake up backend** |
| Chat says "Couldn't reach the backend" | Same as above | Same |
| HTTP 500 `GROQ_API_KEY not set` | Env var missing on **backend** service | Add it in Render → Environment, redeploy |
| Groq error, answers still arrive | Groq rate-limit → automatic Gemini fallback | Normal. Set `GOOGLE_API_KEY` to enable fallback |
| Gemini "model not found" | Google retired the model id | Set `GEMINI_MODEL` to a current id (see https://ai.google.dev/gemini-api/docs/models) |
| "Could not resolve location" | Typo / unknown place | Try "City, State" or "City, Country" |
| Voice: "FFmpeg was not found" | FFmpeg missing locally | Install FFmpeg, restart VS Code. (Docker image already has it) |
| Voice transcription fails on Render | Google Web Speech rate-limit / no Whisper | Use text chat; or run Whisper locally (`pip install openai-whisper`) |
| Build fails: dependency conflict | Old unpinned `langchain-groq` | Use the pinned `requirements-backend.txt` from this repo |

## 5. Security checklist
* The original zip contained a real `.env` — **if it was ever shared or pushed, rotate the Groq and Gemini keys** (Groq console / Google AI Studio) and update them in Render.
* `.env` is in `.gitignore` and `.dockerignore`. Keys live only in Render's Environment tab.
* `ALLOWED_ORIGINS` can restrict CORS to the frontend URL.
