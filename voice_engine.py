"""
voice_engine.py
================================================================================
Module 8: Voice Interaction & Audio Processing for WeatherGPT.

100% free / open-source, no paid API keys:
- STT (speech -> text): local OpenAI Whisper (`tiny`/`base` model, runs on CPU)
  as the primary engine, with `SpeechRecognition` + its free/keyless Google
  Web Speech recognizer as a lightweight fallback if Whisper isn't installed
  or fails to load (e.g. very low-resource machines).
- TTS (text -> speech): `gTTS` (Google Text-to-Speech free Python package,
  no API key -- just hits a public endpoint) as the primary engine, since it
  has solid coverage of Indian languages. `pyttsx3` (fully offline, works
  without internet) is used as a fallback, though its Indian-language voice
  support depends on OS-installed voices and is generally weaker than gTTS.

Both transcribe_audio() and generate_voice_response() are designed to NEVER
hard-crash the app -- on any failure they return (None, error_message) /
(None, error_message) respectively, so app.py can degrade gracefully to
text-only interaction.

Language codes used throughout (BCP-47 / gTTS-style, 2-letter where possible):
    English  -> en   | Hindi    -> hi   | Tamil    -> ta
    Telugu   -> te    | Marathi -> mr   | Bengali  -> bn
    Kannada  -> kn    | Gujarati -> gu
================================================================================
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("weathergpt.voice_engine")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Language configuration
# ------------------------------------------------------------------------------
# Maps our internal short codes -> (Whisper language name, gTTS lang code,
# SpeechRecognition/Google Web Speech BCP-47 locale)
SUPPORTED_VOICE_LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"label": "English",  "whisper": "english",  "gtts": "en", "sr_locale": "en-IN"},
    "hi": {"label": "Hindi",    "whisper": "hindi",    "gtts": "hi", "sr_locale": "hi-IN"},
    "ta": {"label": "Tamil",    "whisper": "tamil",    "gtts": "ta", "sr_locale": "ta-IN"},
    "te": {"label": "Telugu",   "whisper": "telugu",   "gtts": "te", "sr_locale": "te-IN"},
    "mr": {"label": "Marathi",  "whisper": "marathi",  "gtts": "mr", "sr_locale": "mr-IN"},
    "bn": {"label": "Bengali",  "whisper": "bengali",  "gtts": "bn", "sr_locale": "bn-IN"},
    "kn": {"label": "Kannada",  "whisper": "kannada",  "gtts": "kn", "sr_locale": "kn-IN"},
    "gu": {"label": "Gujarati", "whisper": "gujarati", "gtts": "gu", "sr_locale": "gu-IN"},
}

DEFAULT_LANGUAGE = "en"

# Whisper model size -- "tiny" or "base" recommended for CPU/free-tier latency.
# Override with env var WHISPER_MODEL_SIZE if you have more compute available
# (e.g. "small" for better accuracy on Indian languages at the cost of speed).
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

# ------------------------------------------------------------------------------
# FFmpeg configuration
# ------------------------------------------------------------------------------
# Explicitly configure FFmpeg so Whisper and pydub can find it even when
# Streamlit is launched from a process that does not inherit the updated PATH.
# ------------------------------------------------------------------------------
# FFmpeg configuration
# ------------------------------------------------------------------------------
# Use the FFmpeg binaries available on the current operating system.
#
# Local Windows:
#   Uses ffmpeg/ffprobe from PATH if installed.
#
# Render/Linux Docker:
#   Dockerfile.frontend installs ffmpeg using apt-get, so it is available
#   through PATH.

import shutil

FFMPEG_BIN = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg")

FFPROBE_BIN = os.getenv("FFPROBE_BIN") or shutil.which("ffprobe")

@dataclass
class TranscriptionResult:
    text: str
    detected_language: Optional[str] = None   # short code, e.g. "hi"
    engine_used: str = "whisper"


# ------------------------------------------------------------------------------
# Lazy-loaded Whisper model (loading the model is slow; do it once, on first use)
# ------------------------------------------------------------------------------
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    import whisper  # openai-whisper package
    logger.info("Loading local Whisper model (%s)... first call may take a while.", WHISPER_MODEL_SIZE)
    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


# ------------------------------------------------------------------------------
# STT: transcribe_audio
# ------------------------------------------------------------------------------
def transcribe_audio(
    audio_bytes: bytes,
    language_hint: Optional[str] = None,
) -> tuple[Optional[TranscriptionResult], Optional[str]]:
    """
    Convert raw audio bytes (from Streamlit's mic widget -- WAV/MP3/WEBM
    depending on the capture component) into transcribed text.

    Args:
        audio_bytes: Raw audio bytes captured from the browser microphone.
        language_hint: Optional short code (e.g. "hi") to bias/force the
            recognizer's language. If None, Whisper auto-detects the language,
            which works well across the supported Indian languages.

    Returns:
        (TranscriptionResult, None) on success,
        (None, error_message) on failure -- caller should show error_message
        to the user and fall back to text input.
    """
    if not audio_bytes:
        return None, "No audio received from the microphone."

    # ---- Primary engine: local Whisper ----
    try:
        return _transcribe_with_whisper(audio_bytes, language_hint)
    except ImportError:
        logger.warning("openai-whisper not installed; falling back to SpeechRecognition.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper transcription failed (%s); falling back to SpeechRecognition.", exc)

    # ---- Fallback engine: SpeechRecognition (free Google Web Speech, keyless) ----
    try:
        return _transcribe_with_speech_recognition(audio_bytes, language_hint)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Both STT engines failed.")
        return None, f"Speech recognition failed on all available engines: {exc}"


def _transcribe_with_whisper(
    audio_bytes: bytes, language_hint: Optional[str]
) -> tuple[Optional[TranscriptionResult], Optional[str]]:
    model = _get_whisper_model()

    # Whisper's Python API wants a file path (or a preprocessed waveform);
    # writing to a temp file is the simplest robust approach across audio
    # container formats (wav/mp3/webm/ogg) since Whisper uses ffmpeg internally.
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        whisper_lang = None
        if language_hint and language_hint in SUPPORTED_VOICE_LANGUAGES:
            whisper_lang = SUPPORTED_VOICE_LANGUAGES[language_hint]["whisper"]

        result = model.transcribe(tmp_path, language=whisper_lang, fp16=False)
        text = (result.get("text") or "").strip()
        detected = result.get("language")  # Whisper returns e.g. "hi", "en"

        if not text:
            return None, "Whisper could not detect any speech in the recording."

        return TranscriptionResult(text=text, detected_language=detected, engine_used="whisper"), None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _transcribe_with_speech_recognition(
    audio_bytes: bytes, language_hint: Optional[str]
) -> tuple[Optional[TranscriptionResult], Optional[str]]:
    import speech_recognition as sr

    locale = "en-IN"
    if language_hint and language_hint in SUPPORTED_VOICE_LANGUAGES:
        locale = SUPPORTED_VOICE_LANGUAGES[language_hint]["sr_locale"]

    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Ensure WAV format for SpeechRecognition via pydub (handles mp3/webm/ogg too)
        _ensure_wav(audio_bytes, tmp.name)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        # Free, keyless Google Web Speech API (rate-limited, fine for demo use)
        text = recognizer.recognize_google(audio_data, language=locale)
        if not text.strip():
            return None, "No speech detected in the recording."
        return TranscriptionResult(
            text=text.strip(), detected_language=language_hint, engine_used="speech_recognition"
        ), None
    except sr.UnknownValueError:
        return None, "Could not understand the audio -- please try speaking more clearly."
    except sr.RequestError as exc:
        return None, f"Speech recognition service unavailable: {exc}"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

'''
def _ensure_wav(audio_bytes: bytes, out_path: str) -> None:
    """Convert arbitrary browser-captured audio (webm/ogg/mp3) to WAV via pydub/ffmpeg."""
    from pydub import AudioSegment
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    segment.export(out_path, format="wav")
'''
def _ensure_wav(audio_bytes: bytes, out_path: str) -> None:
    """Convert browser audio to WAV using the explicitly configured FFmpeg."""
    from pydub import AudioSegment

    if not os.path.isfile(FFMPEG_BIN):
        raise FileNotFoundError(
            f"FFmpeg executable not found at: {FFMPEG_BIN}"
        )

    AudioSegment.converter = FFMPEG_BIN
    AudioSegment.ffmpeg = FFMPEG_BIN
    AudioSegment.ffprobe = FFPROBE_BIN

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    segment.export(out_path, format="wav")

# ------------------------------------------------------------------------------
# TTS: generate_voice_response
# ------------------------------------------------------------------------------
def generate_voice_response(
    text_response: str,
    language_code: str = DEFAULT_LANGUAGE,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Synthesize the WeatherGPT text answer into spoken audio.

    Args:
        text_response: The final natural-language answer to speak aloud.
        language_code: Short code (e.g. "hi", "ta"). Falls back to English
            if unsupported.

    Returns:
        (mp3_audio_bytes, None) on success,
        (None, error_message) on failure -- caller should fall back to
        text-only display.
    """
    if not text_response or not text_response.strip():
        return None, "No text provided to synthesize."

    lang_info = SUPPORTED_VOICE_LANGUAGES.get(language_code, SUPPORTED_VOICE_LANGUAGES[DEFAULT_LANGUAGE])

    # ---- Primary engine: gTTS (online, free, best Indian-language coverage) ----
    try:
        return _synthesize_with_gtts(text_response, lang_info["gtts"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("gTTS synthesis failed (%s); falling back to pyttsx3 (offline).", exc)

    # ---- Fallback engine: pyttsx3 (fully offline, no internet needed) ----
    try:
        return _synthesize_with_pyttsx3(text_response)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Both TTS engines failed.")
        return None, f"Voice synthesis failed on all available engines: {exc}"


def _synthesize_with_gtts(text: str, gtts_lang: str) -> tuple[Optional[bytes], Optional[str]]:
    from gtts import gTTS

    # gTTS has a practical length limit per request; trim very long responses
    # (e.g. detailed 7-day briefings) to keep audio playback snappy in the demo.
    MAX_CHARS = 800
    trimmed = text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rsplit(".", 1)[0] + "."

    buf = io.BytesIO()
    tts = gTTS(text=trimmed, lang=gtts_lang)
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read(), None


def _synthesize_with_pyttsx3(text: str) -> tuple[Optional[bytes], Optional[str]]:
    import pyttsx3

    # pyttsx3 only supports Indian-language output if the OS has matching
    # voices installed (varies a lot by platform) -- this fallback is mainly
    # useful for English, fully-offline demos with no internet access.
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            return None, "pyttsx3 produced no audio output."
        return audio_bytes, None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    # Manual smoke test: `python voice_engine.py`
    audio, err = generate_voice_response("Rain is expected in Shimla tomorrow evening.", "en")
    if err:
        print("TTS error:", err)
    else:
        with open("test_output.mp3", "wb") as f:
            f.write(audio)
        print("Wrote test_output.mp3 (%d bytes)" % len(audio))
