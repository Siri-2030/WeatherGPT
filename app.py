
"""
app.py
================================================================================
Streamlit Frontend UI for WeatherGPT.

Features:
- Chat-style conversation window (calls the FastAPI /chat endpoint).
- Dynamic Folium map centered on the queried location, with a marker + popup
  showing current conditions.
- Alert badges rendered prominently when the NWP data contains severe-weather
  flags (from wrf_simulator's rule-based alert extraction).
- Module 8: real voice search using streamlit-mic-recorder:
      Browser microphone
          ↓
      streamlit-mic-recorder
          ↓
      voice_engine.transcribe_audio()
          ↓
      Local Whisper / SpeechRecognition fallback
          ↓
      FastAPI /chat
          ↓
      WeatherGPT response
          ↓
      gTTS / pyttsx3
          ↓
      st.audio()
- All voice engines are free/open-source.
- Voice failures degrade gracefully to text-only chat.

Run locally with:
    streamlit run app.py

Backend should already be running on:
    http://localhost:8000
================================================================================
"""

from __future__ import annotations

import os
from typing import Any, Optional

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

import voice_engine


# ==============================================================================
# Configuration
# ==============================================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)


st.set_page_config(
    page_title="WeatherGPT",
    page_icon="⛈️",
    layout="wide",
)


# ==============================================================================
# Microphone recorder
# ==============================================================================

try:
    from streamlit_mic_recorder import mic_recorder

    _HAS_MIC_RECORDER = True

except ImportError:
    _HAS_MIC_RECORDER = False


# ==============================================================================
# Session state initialization
# ==============================================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste! I'm WeatherGPT 🌦️ Ask me about weather, "
                "forecasts, or disaster alerts for any location in "
                "India — in English, Hindi, Tamil, Telugu, and more. "
                "You can type or use the 🎙️ voice search in the sidebar."
            ),
        }
    ]


if "last_nwp_data" not in st.session_state:

    st.session_state.last_nwp_data = None


if "last_answer_audio" not in st.session_state:

    st.session_state.last_answer_audio = None


if "pending_voice_query" not in st.session_state:

    st.session_state.pending_voice_query = None


if "last_processed_audio_id" not in st.session_state:

    st.session_state.last_processed_audio_id = None


# ==============================================================================
# Persona configuration
# ==============================================================================

PERSONA_LABELS = {
    "Auto-detect": None,
    "🌾 Agriculture": "agriculture",
    "🚨 Disaster Management": "disaster",
    "✈️ Aviation": "aviation",
    "⛵ Marine / Coastal": "marine",
    "🔬 Researcher": "researcher",
    "🙂 General Public": "general",
}


ALERT_COLOR = {
    "Severe": "red",
    "Moderate": "orange",
    "Low": "blue",
}


# ==============================================================================
# Voice language configuration
# ==============================================================================

VOICE_LANGUAGE_OPTIONS = {
    code: info["label"]
    for code, info in voice_engine.SUPPORTED_VOICE_LANGUAGES.items()
}


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:

    st.title("⛈️ WeatherGPT")

    st.caption(
        "Smart India Hackathon — Conversational NWP Weather Assistant"
    )

    # --------------------------------------------------------------------------
    # Settings
    # --------------------------------------------------------------------------

    st.markdown("### 🎛️ Settings")

    persona_label = st.selectbox(
        "Advisory mode",
        list(PERSONA_LABELS.keys()),
    )

    default_location = st.text_input(
        "Default location (used if you don't mention one)",
        "Vijayawada",
    )

    # --------------------------------------------------------------------------
    # Voice Search
    # --------------------------------------------------------------------------

    st.markdown("### 🎙️ Voice Search")

    st.caption(
        "Speak your question. Audio is captured using the browser microphone "
        "and transcribed locally with Whisper. A free Google Web Speech "
        "fallback is also available."
    )

    voice_lang_label = st.selectbox(
        "Spoken language",
        list(VOICE_LANGUAGE_OPTIONS.values()),
        help=(
            "Used to bias transcription and to choose "
            "the reply's spoken language."
        ),
    )

    voice_lang_code = next(
        code
        for code, label in VOICE_LANGUAGE_OPTIONS.items()
        if label == voice_lang_label
    )

    enable_tts_playback = st.toggle(
        "🔊 Speak answers aloud",
        value=True,
    )

    captured_audio_bytes: Optional[bytes] = None
    audio_id: Optional[str] = None

    # --------------------------------------------------------------------------
    # Microphone recorder
    # --------------------------------------------------------------------------

    if _HAS_MIC_RECORDER:

        mic_result = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹️ Stop & transcribe",
            just_once=True,
            use_container_width=True,
            key="mic_recorder_widget",
        )

        if mic_result and mic_result.get("bytes"):

            captured_audio_bytes = mic_result["bytes"]

            audio_id = (
                f"mic_{mic_result.get('id', len(captured_audio_bytes))}"
            )

    else:

        st.error(
            "❌ streamlit-mic-recorder is not installed."
        )

        st.info(
            "Install it using: "
            "`pip install streamlit-mic-recorder`"
        )

    # --------------------------------------------------------------------------
    # Transcribe only NEW recordings
    # --------------------------------------------------------------------------

    if (
        captured_audio_bytes
        and audio_id != st.session_state.last_processed_audio_id
    ):

        st.session_state.last_processed_audio_id = audio_id

        with st.spinner("🎙️ Transcribing your question..."):

            transcription, stt_error = (
                voice_engine.transcribe_audio(
                    captured_audio_bytes,
                    language_hint=voice_lang_code,
                )
            )

        if stt_error:

            st.error(
                f"🎙️ Voice transcription failed: {stt_error}"
            )

        elif transcription:

            st.success(
                f'Heard: "{transcription.text}"'
            )

            st.session_state.pending_voice_query = (
                transcription.text
            )

    # --------------------------------------------------------------------------
    # Backend status
    # --------------------------------------------------------------------------

    st.markdown("---")

    try:

        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=3,
        )

        if response.ok:

            backend_status = "🟢 Online"

        else:

            backend_status = "🔴 Offline"

    except requests.RequestException:

        backend_status = (
            "🔴 Offline "
            "(start FastAPI backend on :8000)"
        )

    st.caption(
        f"Backend: {backend_status}"
    )


# ==============================================================================
# Backend /chat API
# ==============================================================================

def call_chat_api(
    message: str,
) -> dict[str, Any]:

    payload = {
        "message": message,
        "default_location": default_location or None,
        "persona": PERSONA_LABELS.get(persona_label),
    }

    try:

        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:

        return {
            "answer": (
                f"⚠️ Couldn't reach the WeatherGPT backend "
                f"at `{BACKEND_URL}`. "
                f"Is `main.py` (FastAPI) running? "
                f"Details: {exc}"
            ),
            "location_used": None,
            "nwp_data": None,
            "persona": "general",
            "error": str(exc),
        }


# ==============================================================================
# Alert badges
# ==============================================================================

def render_alert_badges(
    nwp_data: dict[str, Any] | None,
) -> None:

    if not nwp_data:

        return

    alerts = nwp_data.get("alerts") or []

    if not alerts:

        return

    st.markdown(
        "#### 🚨 Active Weather Alerts"
    )

    cols = st.columns(len(alerts))

    for col, alert in zip(cols, alerts):

        color = ALERT_COLOR.get(
            alert.get("severity", "Moderate"),
            "orange",
        )

        with col:

            st.markdown(
                f"""
                <div style="
                    border:2px solid {color};
                    border-radius:10px;
                    padding:10px;
                    background-color:rgba(255,0,0,0.05);
                ">
                    <b style="color:{color};">
                        {alert['type']} — {alert['severity']}
                    </b>
                    <br>
                    <span style="font-size:0.9em;">
                        {alert['message']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# Weather map
# ==============================================================================

def render_map(
    nwp_data: dict[str, Any] | None,
) -> None:

    if not nwp_data or not nwp_data.get("location"):

        st.info(
            "Ask about a location to see it on the map."
        )

        return

    loc = nwp_data["location"]

    current = nwp_data.get(
        "current",
        {},
    )

    m = folium.Map(
        location=[
            loc["lat"],
            loc["lon"],
        ],
        zoom_start=8,
        tiles="CartoDB positron",
    )

    popup_html = (
        f"<b>{loc['display_name']}</b><br>"
        f"🌡️ {current.get('temperature_2m', '—')}°C &nbsp; "
        f"💧 {current.get('relative_humidity_2m', '—')}%<br>"
        f"💨 Wind {current.get('wind_speed_10m', '—')} km/h "
        f"(gusts {current.get('wind_gusts_10m', '—')} km/h)"
    )

    folium.Marker(
        [
            loc["lat"],
            loc["lon"],
        ],
        popup=folium.Popup(
            popup_html,
            max_width=250,
        ),
        tooltip=loc["display_name"],
        icon=folium.Icon(
            color="blue",
            icon="cloud",
        ),
    ).add_to(m)

    for alert in nwp_data.get("alerts", []):

        color = ALERT_COLOR.get(
            alert.get("severity", "Moderate"),
            "orange",
        )

        folium.Circle(
            [
                loc["lat"],
                loc["lon"],
            ],
            radius=15000,
            color=color,
            fill=True,
            fill_opacity=0.15,
        ).add_to(m)

    st_folium(
        m,
        width=None,
        height=380,
        key="weather_map",
    )


# ==============================================================================
# Process user query
# ==============================================================================

def process_user_query(
    query: str,
) -> None:

    """
    Shared pipeline for typed and voice queries.

    1. Add user query to conversation.
    2. Send query to FastAPI backend.
    3. Display WeatherGPT response.
    4. Generate spoken response if TTS is enabled.
    """

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with chat_container:

        with st.chat_message("user"):

            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner(
                "Fetching live NWP data & thinking…"
            ):

                result = call_chat_api(query)

            st.markdown(
                result["answer"]
            )

            answer_audio = None

            # ------------------------------------------------------------------
            # Text-to-speech
            # ------------------------------------------------------------------

            if (
                enable_tts_playback
                and result.get("answer")
                and not result.get("error")
            ):

                with st.spinner(
                    "🔊 Synthesizing voice reply..."
                ):

                    (
                        answer_audio,
                        tts_error,
                    ) = voice_engine.generate_voice_response(
                        result["answer"],
                        language_code=voice_lang_code,
                    )

                if tts_error:

                    st.caption(
                        f"🔇 Voice reply unavailable: {tts_error}"
                    )

                elif answer_audio:

                    st.audio(
                        answer_audio,
                        format="audio/mp3",
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
        }
    )

    st.session_state.last_nwp_data = (
        result.get("nwp_data")
    )

    st.session_state.last_answer_audio = (
        answer_audio
    )


# ==============================================================================
# Main layout
# ==============================================================================

chat_col, map_col = st.columns(
    [1.3, 1]
)


# ==============================================================================
# Chat column
# ==============================================================================

with chat_col:

    st.subheader(
        "💬 Conversation"
    )

    chat_container = st.container(
        height=480
    )

    with chat_container:

        for msg in st.session_state.messages:

            with st.chat_message(
                msg["role"]
            ):

                st.markdown(
                    msg["content"]
                )

    typed_input = st.chat_input(
        "Ask about weather, forecasts, or disaster alerts…"
    )

    # --------------------------------------------------------------------------
    # Voice query
    # --------------------------------------------------------------------------

    voice_query = (
        st.session_state.pending_voice_query
    )

    if voice_query:

        st.session_state.pending_voice_query = None

    final_query = (
        typed_input
        or voice_query
    )

    if final_query:

        process_user_query(
            final_query
        )


# ==============================================================================
# Map and alerts column
# ==============================================================================

with map_col:

    st.subheader(
        "🗺️ Location & Alerts"
    )

    render_alert_badges(
        st.session_state.last_nwp_data
    )

    render_map(
        st.session_state.last_nwp_data
    )

    if st.session_state.last_nwp_data:

        with st.expander(
            "📊 Raw NWP data (debug)"
        ):

            st.json(
                st.session_state.last_nwp_data
            )


# ==============================================================================
# Footer
# ==============================================================================

st.markdown("---")

st.caption(
    "Data: Open-Meteo GFS API (NOAA GFS numerical weather model) · "
    "Geocoding: OpenStreetMap Nominatim · "
    "Voice: local Whisper / gTTS · "
    "Built for Smart India Hackathon — 100% open-source stack."
)

