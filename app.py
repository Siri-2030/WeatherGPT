"""
app.py
================================================================================
Streamlit frontend for WeatherGPT.

Tabs
----
💬 Assistant  - chat (typed or voice) + live map + alert badges
📊 Dashboard  - current-conditions metrics, 24h and 7-day charts, air quality
ℹ️ About      - architecture, data sources, tech stack (useful for the demo)

Voice pipeline
--------------
Browser mic -> streamlit-mic-recorder -> voice_engine.transcribe_audio()
  -> FastAPI /chat -> answer -> voice_engine.generate_voice_response() -> st.audio

Voice is OPTIONAL: if the mic component cannot load (e.g. PyArrow problem,
missing package), the app shows a notice and keeps working as a text chatbot.

Run locally:   streamlit run app.py      (backend on http://localhost:8000)
================================================================================
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Optional

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

import voice_engine

# ==============================================================================
# Configuration
# ==============================================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="WeatherGPT", page_icon="⛈️", layout="wide")

try:
    from streamlit_mic_recorder import mic_recorder

    _HAS_MIC_RECORDER = True
except Exception:  # ImportError, or any import-time failure of the component
    mic_recorder = None  # type: ignore[assignment]
    _HAS_MIC_RECORDER = False

PERSONA_LABELS = {
    "Auto-detect": None,
    "🌾 Agriculture": "agriculture",
    "🚨 Disaster Management": "disaster",
    "✈️ Aviation": "aviation",
    "⛵ Marine / Coastal": "marine",
    "🔬 Researcher": "researcher",
    "🙂 General Public": "general",
}
ALERT_COLOR = {"Severe": "red", "Moderate": "orange", "Low": "blue"}
VOICE_LANGUAGE_OPTIONS = {
    code: info["label"] for code, info in voice_engine.SUPPORTED_VOICE_LANGUAGES.items()
}

EXAMPLE_QUERIES = [
    "Will it rain in Vijayawada tomorrow?",
    "What's the weather in Mumbai right now?",
    "Is it safe for fishermen to go out near Chennai?",
    "Should farmers in Punjab irrigate this week?",
    "హైదరాబాద్ లో రేపు వాతావరణం ఎలా ఉంటుంది?",
    "दिल्ली में आज हवा की गुणवत्ता कैसी है?",
]

GREETING = (
    "Namaste! I'm WeatherGPT 🌦️ Ask me about weather, forecasts, air quality or "
    "disaster alerts for any location — in English, Hindi, Tamil, Telugu and more. "
    "Type below, tap an example, or use 🎙️ voice search in the sidebar."
)

# ==============================================================================
# Session state
# ==============================================================================
_defaults: dict[str, Any] = {
    "messages": [{"role": "assistant", "content": GREETING}],
    "last_nwp_data": None,
    "last_meta": None,          # {"persona":..., "latency_ms":..., "location_used":...}
    "pending_voice_query": None,
    "queued_query": None,
    "last_processed_audio_id": None,
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)


# ==============================================================================
# Backend helpers
# ==============================================================================
@st.cache_data(ttl=20, show_spinner=False)
def get_backend_health(url: str) -> Optional[dict[str, Any]]:
    """Cached /health probe (so every Streamlit rerun doesn't block on it)."""
    try:
        r = requests.get(f"{url}/health", timeout=8)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def wake_backend(max_wait: int = 90) -> bool:
    """Render's free tier sleeps after ~15 min idle; poll /health until it wakes."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            if requests.get(f"{BACKEND_URL}/health", timeout=10).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def call_chat_api(message: str, default_location: str, persona_label: str) -> dict[str, Any]:
    """POST /chat with recent history; retries once to ride out a cold start."""
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[1:-1]  # skip greeting and the message just added
    ][-6:]
    payload = {
        "message": message,
        "default_location": default_location or None,
        "persona": PERSONA_LABELS.get(persona_label),
        "history": history,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
            if r.status_code >= 400:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                return _error_result(f"Backend returned HTTP {r.status_code}: {detail}")
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == 0:
                wake_backend(60)  # cold start: wait for it, then retry once
        except requests.RequestException as exc:
            last_exc = exc
            break
    return _error_result(
        f"Couldn't reach the WeatherGPT backend at `{BACKEND_URL}`. "
        f"Check that it is running and that BACKEND_URL is correct. Details: {last_exc}"
    )


def _error_result(message: str) -> dict[str, Any]:
    return {
        "answer": f"⚠️ {message}",
        "location_used": None,
        "nwp_data": None,
        "persona": "general",
        "error": message,
        "latency_ms": None,
    }


# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.title("⛈️ WeatherGPT")
    st.caption("Conversational NWP weather assistant · multilingual · voice enabled")

    st.markdown("### 🎛️ Settings")
    persona_label = st.selectbox("Advisory mode", list(PERSONA_LABELS.keys()))
    default_location = st.text_input("Default location (used if you don't mention one)", "Vijayawada")

    # ---- Voice ----------------------------------------------------------------
    st.markdown("### 🎙️ Voice Search")
    st.caption(
        "Speak your question. Audio is captured in the browser and transcribed "
        "with Whisper (if installed) or the free Google Web Speech fallback."
    )
    voice_lang_label = st.selectbox(
        "Spoken language",
        list(VOICE_LANGUAGE_OPTIONS.values()),
        help="Used to bias transcription and to choose the language of the spoken reply.",
    )
    voice_lang_code = next(c for c, lbl in VOICE_LANGUAGE_OPTIONS.items() if lbl == voice_lang_label)
    enable_tts_playback = st.toggle("🔊 Speak answers aloud", value=True)

    captured_audio: Optional[bytes] = None
    if _HAS_MIC_RECORDER:
        try:
            mic_result = mic_recorder(
                start_prompt="🎙️ Start recording",
                stop_prompt="⏹️ Stop & transcribe",
                just_once=True,
                use_container_width=True,
                key="mic_recorder_widget",
            )
            if mic_result and mic_result.get("bytes"):
                captured_audio = mic_result["bytes"]
        except Exception as exc:  # noqa: BLE001 - e.g. the PyArrow StreamlitAPIException
            st.warning(
                "🎙️ Voice input is unavailable right now (the microphone component "
                "failed to load). Text chat still works."
            )
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
    else:
        st.warning("🎙️ Voice input unavailable: `streamlit-mic-recorder` is not installed.")

    # Transcribe only NEW recordings
    if captured_audio:
        audio_id = hashlib.md5(captured_audio).hexdigest()
        if audio_id != st.session_state.last_processed_audio_id:
            st.session_state.last_processed_audio_id = audio_id
            with st.spinner("🎙️ Transcribing your question..."):
                transcription, stt_error = voice_engine.transcribe_audio(
                    captured_audio, language_hint=voice_lang_code
                )
            if stt_error:
                st.error(f"🎙️ Voice transcription failed: {stt_error}")
            elif transcription:
                st.success(f'Heard: "{transcription.text}"')
                st.session_state.pending_voice_query = transcription.text

    # ---- Backend status ------------------------------------------------------
    st.markdown("---")
    health = get_backend_health(BACKEND_URL)
    if health:
        llm = health.get("llm", {})
        st.caption(f"Backend: 🟢 Online · v{health.get('version', '?')}")
        st.caption(f"LLM: `{llm.get('provider', '?')}` / `{llm.get('model', '?')}`")
    else:
        st.caption("Backend: 🔴 Offline or waking up")
        st.caption("Render's free tier sleeps when idle; the first request can take ~1 minute.")
        if st.button("⏰ Wake up backend", use_container_width=True):
            with st.spinner("Waking the backend (up to ~90 s)..."):
                ok = wake_backend(90)
            get_backend_health.clear()
            if ok:
                st.rerun()
            else:
                st.error("Still not reachable. Check the Render logs.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.session_state.last_nwp_data = None
        st.session_state.last_meta = None
        st.rerun()
    st.caption(f"Backend URL: `{BACKEND_URL}`")


# ==============================================================================
# Rendering helpers
# ==============================================================================
def render_alert_badges(nwp_data: Optional[dict[str, Any]]) -> None:
    alerts = (nwp_data or {}).get("alerts") or []
    if not alerts:
        if nwp_data:
            st.success("✅ No severe-weather alerts for the next 24 hours.")
        return
    st.markdown("#### 🚨 Active Weather Alerts")
    for alert in alerts:
        color = ALERT_COLOR.get(alert.get("severity", "Moderate"), "orange")
        st.markdown(
            f"""
            <div style="border:2px solid {color};border-radius:10px;padding:10px;
                        margin-bottom:8px;background-color:rgba(255,0,0,0.05);">
              <b style="color:{color};">{alert.get('type')} — {alert.get('severity')}</b><br>
              <span style="font-size:0.9em;">{alert.get('message')}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def render_map(nwp_data: Optional[dict[str, Any]]) -> None:
    if not nwp_data or not nwp_data.get("location"):
        st.info("Ask about a location to see it on the map.")
        return
    loc, cur = nwp_data["location"], nwp_data.get("current", {})
    fmap = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=8, tiles="CartoDB positron")
    popup_html = (
        f"<b>{loc['display_name']}</b><br>"
        f"{cur.get('weather_emoji', '')} {cur.get('weather_description', '')}<br>"
        f"🌡️ {cur.get('temperature_2m', '—')}°C &nbsp; 💧 {cur.get('relative_humidity_2m', '—')}%<br>"
        f"💨 Wind {cur.get('wind_speed_10m', '—')} km/h (gusts {cur.get('wind_gusts_10m', '—')} km/h)"
    )
    folium.Marker(
        [loc["lat"], loc["lon"]],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=loc["display_name"],
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(fmap)
    for alert in nwp_data.get("alerts", []):
        folium.Circle(
            [loc["lat"], loc["lon"]],
            radius=15000,
            color=ALERT_COLOR.get(alert.get("severity", "Moderate"), "orange"),
            fill=True,
            fill_opacity=0.15,
        ).add_to(fmap)
    # returned_objects=[] stops pan/zoom on the map from re-running the whole app
    st_folium(fmap, use_container_width=True, height=360, key="weather_map", returned_objects=[])


def _fmt(value: Any, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{unit}"
    except (TypeError, ValueError):
        return str(value)


def render_dashboard(nwp_data: Optional[dict[str, Any]]) -> None:
    if not nwp_data:
        st.info("Ask the assistant about a location first — its data will appear here.")
        return

    loc, cur = nwp_data["location"], nwp_data.get("current", {})
    st.subheader(f"{cur.get('weather_emoji', '')} {loc['display_name']}")
    st.caption(
        f"{cur.get('weather_description', '')} · local time {nwp_data.get('local_time', '—')} "
        f"({nwp_data.get('timezone', '—')}) · source: {nwp_data.get('source', '')}"
    )

    aq = nwp_data.get("air_quality") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ Temperature", _fmt(cur.get("temperature_2m"), " °C"))
    c2.metric("💧 Humidity", _fmt(cur.get("relative_humidity_2m"), " %", 0))
    c3.metric("💨 Wind", _fmt(cur.get("wind_speed_10m"), " km/h"), f"gusts {_fmt(cur.get('wind_gusts_10m'), '', 0)}", delta_color="off")
    c4.metric("🌧️ Rain now", _fmt(cur.get("precipitation"), " mm"))
    c5.metric("🫁 US AQI", _fmt(aq.get("us_aqi"), "", 0), aq.get("category", "n/a"), delta_color="off")

    hourly = pd.DataFrame(nwp_data.get("hourly_next_24h") or [])
    daily = pd.DataFrame(nwp_data.get("daily_7day") or [])
    # JSON nulls -> NaN so the charts never choke on all-None (object) columns
    for frame in (hourly, daily):
        for col in frame.columns:
            if col not in ("time", "date"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Next 24 hours — temperature (°C)")
        if not hourly.empty:
            # "2026-11-13T07:00+0530" -> "07:00" (already LOCAL time from the backend)
            hdf = hourly.assign(time=hourly["time"].astype(str).str[11:16])
            st.line_chart(hdf.set_index("time")[["temperature_2m"]])
            st.markdown("##### Next 24 hours — rain probability (%) & precipitation (mm)")
            st.bar_chart(hdf.set_index("time")[["precipitation_probability", "precipitation"]])
    with right:
        st.markdown("##### 7-day outlook — max / min temperature (°C)")
        if not daily.empty:
            ddf = daily.set_index("date")
            st.line_chart(ddf[["temperature_2m_max", "temperature_2m_min"]])
            st.markdown("##### 7-day outlook — rainfall (mm)")
            st.bar_chart(ddf[["precipitation_sum"]])

    if not daily.empty:
        with st.expander("📅 Daily forecast table"):
            st.dataframe(daily, use_container_width=True, hide_index=True)
    if aq:
        with st.expander("🫁 Air quality details"):
            st.json(aq)


def render_about() -> None:
    st.subheader("About WeatherGPT")
    st.markdown(
        """
**WeatherGPT** is a conversational AI weather assistant that answers in the
user's own language using *real numerical-weather-prediction (NWP) data*.

```text
User (text / voice, any Indian language)
        │
        ▼
Streamlit UI (app.py) ── mic → Whisper / Google Web Speech (voice_engine.py)
        │  HTTP POST /chat (+ conversation history)
        ▼
FastAPI backend (main.py)
        │
        ▼
LLM router (llm_router.py)
   1. LLM extracts location + forecast horizon (JSON)
   2. Geocode + fetch NWP data (wrf_simulator.py → Open-Meteo / NOAA GFS)
   3. Persona prompt + live JSON → Groq (Gemini fallback) → multilingual answer
        │
        ▼
Answer + map + alerts + charts + spoken reply (gTTS)
```

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Folium map, native charts |
| Backend | FastAPI + Uvicorn |
| LLM | Groq (primary) · Google Gemini (fallback) · Ollama (offline option) via LangChain |
| Weather data | Open-Meteo GFS API (NOAA GFS), Open-Meteo Air Quality |
| Geocoding | Open-Meteo Geocoding API |
| Voice | streamlit-mic-recorder · Whisper / SpeechRecognition · gTTS / pyttsx3 |
| Deployment | Docker, Render (backend + frontend as two services) |

**Advisory personas:** general public, agriculture, disaster management,
aviation, marine/coastal and researcher — each with its own tone and focus.

**Rule-based alerts:** high wind, thunderstorm (CAPE), heavy rainfall, heat wave
and poor air quality. These are simplified demo thresholds, *not* official
warnings — always follow IMD advisories in a real emergency.
        """
    )


# ==============================================================================
# Query pipeline
# ==============================================================================
def process_user_query(query: str, container) -> None:
    """Shared by typed, voice and example-button queries."""
    st.session_state.messages.append({"role": "user", "content": query})

    answer_audio: Optional[bytes] = None
    with container:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Fetching live NWP data & thinking…"):
                result = call_chat_api(query, default_location, persona_label)
            st.markdown(result["answer"])

            if enable_tts_playback and result.get("answer") and not result.get("error"):
                with st.spinner("🔊 Synthesizing voice reply..."):
                    answer_audio, tts_error = voice_engine.generate_voice_response(
                        result["answer"], language_code=voice_lang_code
                    )
                if tts_error:
                    st.caption(f"🔇 Voice reply unavailable: {tts_error}")
                elif answer_audio:
                    st.audio(answer_audio, format=voice_engine.audio_mime(answer_audio))

            meta_bits = []
            if result.get("location_used"):
                meta_bits.append(f"📍 {result['location_used']}")
            if result.get("persona"):
                meta_bits.append(f"🎭 {result['persona']}")
            if result.get("latency_ms") is not None:
                meta_bits.append(f"⏱️ {result['latency_ms'] / 1000:.1f}s")
            if meta_bits:
                st.caption(" · ".join(meta_bits))

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    # Keep the previous map/data if this turn produced none (e.g. an error)
    if result.get("nwp_data"):
        st.session_state.last_nwp_data = result["nwp_data"]
    st.session_state.last_meta = {k: result.get(k) for k in ("persona", "latency_ms", "location_used")}


# ==============================================================================
# Main layout
# ==============================================================================
tab_chat, tab_dash, tab_about = st.tabs(["💬 Assistant", "📊 Dashboard", "ℹ️ About"])

with tab_chat:
    chat_col, map_col = st.columns([1.3, 1])

    with chat_col:
        st.subheader("💬 Conversation")

        st.caption("Try an example:")
        ex_cols = st.columns(3)
        for i, example in enumerate(EXAMPLE_QUERIES):
            if ex_cols[i % 3].button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.queued_query = example

        chat_container = st.container(height=440)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        typed_input = st.chat_input("Ask about weather, forecasts, air quality or disaster alerts…")

        voice_query = st.session_state.pending_voice_query
        st.session_state.pending_voice_query = None
        queued = st.session_state.queued_query
        st.session_state.queued_query = None

        final_query = typed_input or voice_query or queued
        if final_query:
            process_user_query(final_query, chat_container)

    with map_col:
        st.subheader("🗺️ Location & Alerts")
        render_alert_badges(st.session_state.last_nwp_data)
        render_map(st.session_state.last_nwp_data)
        if st.session_state.last_nwp_data:
            with st.expander("📊 Raw NWP data (debug)"):
                st.json(st.session_state.last_nwp_data)

with tab_dash:
    render_dashboard(st.session_state.last_nwp_data)

with tab_about:
    render_about()

st.markdown("---")
st.caption(
    "Data: Open-Meteo GFS API (NOAA GFS) · Geocoding: Open-Meteo · "
    "Voice: Whisper / Google Web Speech / gTTS · 100% free & open-source stack."
)
