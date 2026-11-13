"""
wrf_simulator.py
================================================================================
Data Ingestor Module for WeatherGPT.

NOTE ON NAMING: this module is called `wrf_simulator` to match the project
architecture, but it does NOT run a real WRF (Weather Research and Forecasting)
binary. WRF needs a compiled Fortran/MPI model and HPC hardware, which free
hosting (Render / Hugging Face Spaces) cannot provide.

Instead it treats the Open-Meteo API as a hosted, pre-computed NWP proxy:
Open-Meteo re-serves the same NOAA GFS model output that a self-hosted WRF run
would be initialised from -- free, keyless and fast.

Responsibilities
----------------
1. Geocode a free-text place name -> (lat, lon) using Open-Meteo Geocoding.
2. Fetch current + hourly + daily NWP variables from the Open-Meteo GFS API.
3. Add air quality (Open-Meteo Air Quality API) and rule-based alerts.
4. Return a clean, JSON-safe dict ready for LLM prompt injection / the UI.

Fixes in this version (vs. the first deployment)
------------------------------------------------
* NaN / Inf values are converted to None. FastAPI refuses to serialise NaN, so a
  single missing GFS value used to make /chat and /weather return HTTP 500.
* "hourly_next_24h" now starts at the CURRENT local hour (it used to start at
  00:00 UTC of the first forecast day) and timestamps are in LOCAL time.
* "City, State" queries ("Shimla, Himachal Pradesh") now geocode correctly.
  Open-Meteo only matches the place name, so we search the first part and use
  the remaining parts to pick the right candidate.
* The HTTP cache lives in a writable temp dir and falls back to an in-memory
  cache, so a read-only filesystem can no longer crash the app at import time.
* Added: weather-code descriptions/emojis, air quality, heat-wave and poor-air
  alerts, local timezone info.
================================================================================
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry

logger = logging.getLogger("weathergpt.wrf_simulator")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------
# HTTP session: cached + retrying
# ------------------------------------------------------------------------------
def _build_session() -> requests.Session:
    """Create a cached session; never crash the app if the cache can't be used."""
    cache_path = os.getenv(
        "WEATHERGPT_CACHE_PATH",
        os.path.join(tempfile.gettempdir(), "weathergpt_http_cache"),
    )
    try:
        return requests_cache.CachedSession(cache_path, expire_after=900)
    except Exception as exc:  # noqa: BLE001 - read-only FS, locked sqlite, ...
        logger.warning("SQLite HTTP cache unavailable (%s); using in-memory cache.", exc)
        return requests_cache.CachedSession(backend="memory", expire_after=900)


_cache_session = _build_session()
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
_openmeteo = openmeteo_requests.Client(session=_retry_session)


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/gfs"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# ------------------------------------------------------------------------------
# NWP variables (the ORDER matters: the SDK returns variables by index)
# ------------------------------------------------------------------------------
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "rain",
    "showers",
    "weather_code",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cape",
    "soil_moisture_0_to_1cm",
    "soil_temperature_0cm",
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "uv_index_max",
]

CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cape",
]

AIR_QUALITY_VARS = ["pm2_5", "pm10", "us_aqi", "ozone", "carbon_monoxide"]


# ------------------------------------------------------------------------------
# WMO weather code -> (description, emoji)
# ------------------------------------------------------------------------------
WEATHER_CODES: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Light freezing drizzle", "🌧️"),
    57: ("Dense freezing drizzle", "🌧️"),
    61: ("Slight rain", "🌦️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Slight snow", "🌨️"),
    73: ("Moderate snow", "🌨️"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Slight snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}


def describe_weather_code(code: Optional[float]) -> tuple[str, str]:
    """Return (description, emoji) for a WMO code; safe for None / unknown."""
    if code is None:
        return "Unknown", "❔"
    return WEATHER_CODES.get(int(code), ("Unknown", "❔"))


def aqi_category(us_aqi: Optional[float]) -> str:
    """US EPA AQI category label."""
    if us_aqi is None:
        return "Unknown"
    if us_aqi <= 50:
        return "Good"
    if us_aqi <= 100:
        return "Moderate"
    if us_aqi <= 150:
        return "Unhealthy for sensitive groups"
    if us_aqi <= 200:
        return "Unhealthy"
    if us_aqi <= 300:
        return "Very unhealthy"
    return "Hazardous"


# ------------------------------------------------------------------------------
# JSON safety
# ------------------------------------------------------------------------------
def sanitize(obj: Any) -> Any:
    """
    Recursively make `obj` JSON-compliant: NaN/Inf -> None, numpy scalars ->
    Python scalars. (FastAPI/Starlette raise ValueError on NaN.)
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return sanitize(obj.item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------
@dataclass
class GeoLocation:
    query: str
    lat: float
    lon: float
    display_name: str
    country: Optional[str] = None
    state: Optional[str] = None


@dataclass
class NWPResult:
    location: GeoLocation
    fetched_at_utc: str
    timezone: str = "UTC"
    local_time: Optional[str] = None
    current: dict[str, Any] = field(default_factory=dict)
    hourly_next_24h: list[dict[str, Any]] = field(default_factory=list)
    daily_7day: list[dict[str, Any]] = field(default_factory=list)
    air_quality: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, str]] = field(default_factory=list)
    source: str = "Open-Meteo GFS API (NOAA GFS 0.25deg proxy)"

    def to_dict(self) -> dict[str, Any]:
        return sanitize(
            {
                "location": {
                    "query": self.location.query,
                    "lat": self.location.lat,
                    "lon": self.location.lon,
                    "display_name": self.location.display_name,
                    "country": self.location.country,
                    "state": self.location.state,
                },
                "fetched_at_utc": self.fetched_at_utc,
                "timezone": self.timezone,
                "local_time": self.local_time,
                "current": self.current,
                "hourly_next_24h": self.hourly_next_24h,
                "daily_7day": self.daily_7day,
                "air_quality": self.air_quality,
                "alerts": self.alerts,
                "source": self.source,
            }
        )


# ------------------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------------------
class GeocodingError(Exception):
    pass


class NWPFetchError(Exception):
    pass


# ------------------------------------------------------------------------------
# Geocoding
# ------------------------------------------------------------------------------
# Pre-seeded coordinates for the demo's default city (saves one API call).
_LOCATION_CACHE: dict[str, GeoLocation] = {
    "vijayawada": GeoLocation(
        query="Vijayawada", lat=16.5062, lon=80.6480,
        display_name="Vijayawada, Andhra Pradesh, India",
        country="India", state="Andhra Pradesh",
    ),
    "vijayawada, andhra pradesh": GeoLocation(
        query="Vijayawada, Andhra Pradesh", lat=16.5062, lon=80.6480,
        display_name="Vijayawada, Andhra Pradesh, India",
        country="India", state="Andhra Pradesh",
    ),
}


def _search_places(name: str, count: int = 10) -> list[dict[str, Any]]:
    """Call the Open-Meteo geocoding endpoint; returns the raw result list."""
    try:
        resp = _retry_session.get(
            GEOCODING_URL,
            params={"name": name, "count": count, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("results") or []
    except requests.RequestException as exc:
        raise GeocodingError(f"Geocoding service unavailable: {exc}") from exc


def _pick_best_match(results: list[dict[str, Any]], hints: list[str]) -> dict[str, Any]:
    """
    Choose the candidate whose admin1/admin2/country matches the extra words
    the user typed (e.g. "Himachal Pradesh", "India"). Falls back to the
    top-ranked result.
    """
    hints = [h.strip().lower() for h in hints if h.strip()]
    if not hints:
        return results[0]

    def score(r: dict[str, Any]) -> int:
        haystack = " ".join(
            str(r.get(k) or "").lower()
            for k in ("admin1", "admin2", "admin3", "country", "country_code")
        )
        return sum(1 for h in hints if h in haystack)

    best = max(results, key=score)  # max() keeps the first (best-ranked) on ties
    return best if score(best) > 0 else results[0]


def geocode_location(place_name: str) -> GeoLocation:
    """Convert a free-text place name into coordinates (worldwide)."""
    normalized = " ".join(place_name.strip().lower().split())
    if not normalized:
        raise GeocodingError("Location name cannot be empty.")

    if normalized in _LOCATION_CACHE:
        cached = _LOCATION_CACHE[normalized]
        return GeoLocation(
            query=place_name, lat=cached.lat, lon=cached.lon,
            display_name=cached.display_name, country=cached.country, state=cached.state,
        )

    # Open-Meteo matches the *name* only, so "Shimla, Himachal Pradesh" returns
    # nothing. Search the first comma-separated part; use the rest as hints.
    parts = [p.strip() for p in place_name.split(",") if p.strip()]
    primary, hints = parts[0], parts[1:]

    results = _search_places(primary)
    if not results and hints:  # last resort: maybe the full string was a name
        results = _search_places(place_name.strip())
        hints = []
    if not results:
        raise GeocodingError(f"Could not resolve location: '{place_name}'")

    match = _pick_best_match(results, hints)
    latitude, longitude = match.get("latitude"), match.get("longitude")
    if latitude is None or longitude is None:
        raise GeocodingError(f"Geocoding returned incomplete coordinates for '{place_name}'")

    name = match.get("name") or primary
    admin1 = match.get("admin1")
    country = match.get("country")
    display = [name]
    if admin1 and admin1.lower() != name.lower():
        display.append(admin1)
    if country:
        display.append(country)

    location = GeoLocation(
        query=place_name,
        lat=round(float(latitude), 4),
        lon=round(float(longitude), 4),
        display_name=", ".join(display),
        country=country,
        state=admin1,
    )
    _LOCATION_CACHE[normalized] = location
    logger.info("Geocoded '%s' -> %s (%s, %s)", place_name, location.display_name, location.lat, location.lon)
    return location


# ------------------------------------------------------------------------------
# Alert extraction (simplified demo thresholds -- NOT official warnings)
# ------------------------------------------------------------------------------
def _extract_current_alerts(
    next_24h: pd.DataFrame,
    air_quality: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if next_24h.empty:
        return alerts

    def _add(kind: str, severity: str, message: str) -> None:
        alerts.append({"type": kind, "severity": severity, "message": message})

    max_gust = next_24h["wind_gusts_10m"].max()
    if pd.notna(max_gust) and max_gust >= 60:
        _add("High Wind", "Severe" if max_gust >= 90 else "Moderate",
             f"Wind gusts up to {max_gust:.0f} km/h expected in the next 24h.")

    max_cape = next_24h["cape"].max()
    if pd.notna(max_cape) and max_cape >= 1500:
        _add("Thunderstorm / Severe Convection Risk", "Severe" if max_cape >= 2500 else "Moderate",
             f"High CAPE ({max_cape:.0f} J/kg) indicates strong potential for thunderstorms/hail.")

    total_rain = next_24h["precipitation"].sum()
    if pd.notna(total_rain) and total_rain >= 60:
        _add("Heavy Rainfall / Flood Risk", "Severe" if total_rain >= 115 else "Moderate",
             f"Cumulative rainfall of {total_rain:.0f} mm expected in next 24h.")

    max_temp = next_24h["temperature_2m"].max()
    if pd.notna(max_temp) and max_temp >= 40:
        _add("Heat Wave", "Severe" if max_temp >= 45 else "Moderate",
             f"Temperatures up to {max_temp:.0f}°C expected in the next 24h.")

    aqi = (air_quality or {}).get("us_aqi")
    if aqi is not None and aqi >= 151:
        _add("Poor Air Quality", "Severe" if aqi >= 201 else "Moderate",
             f"US AQI is {aqi:.0f} ({aqi_category(aqi)}). Limit outdoor exertion.")

    return alerts


# ------------------------------------------------------------------------------
# Air quality
# ------------------------------------------------------------------------------
def fetch_air_quality(lat: float, lon: float) -> dict[str, Any]:
    """Current air quality from the free Open-Meteo Air Quality API ({} on failure)."""
    params = {"latitude": lat, "longitude": lon, "current": AIR_QUALITY_VARS}
    try:
        responses = _openmeteo.weather_api(OPEN_METEO_AIR_QUALITY_URL, params=params)
        current = responses[0].Current()
        values = {k: round(float(current.Variables(i).Value()), 1) for i, k in enumerate(AIR_QUALITY_VARS)}
        values = sanitize(values)
        values["category"] = aqi_category(values.get("us_aqi"))
        return values
    except Exception as exc:  # noqa: BLE001
        logger.warning("Air quality fetch failed: %s", exc)
        return {}


# ------------------------------------------------------------------------------
# Main NWP fetch
# ------------------------------------------------------------------------------
def fetch_nwp_data(
    place_name: str,
    forecast_days: int = 7,
    include_air_quality: bool = True,
) -> NWPResult:
    """
    Main entry point: geocode -> fetch NWP -> structure -> alerts.

    HPC-UPGRADE-PATH: to use real WRF output, replace the Open-Meteo call with
    a parser for wrfout_*.nc (T2, RAINNC, U10, V10, ...) that fills the SAME
    `current`, `hourly_next_24h` and `daily_7day` structures. llm_router.py and
    prompts.py then stay unchanged.
    """
    location = geocode_location(place_name)

    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "hourly": HOURLY_VARS,
        "daily": DAILY_VARS,
        "current": CURRENT_VARS,
        "forecast_days": min(max(int(forecast_days), 1), 16),
        "timezone": "auto",  # daily values are computed for the LOCAL day
    }

    try:
        response = _openmeteo.weather_api(OPEN_METEO_FORECAST_URL, params=params)[0]
    except Exception as exc:  # noqa: BLE001
        raise NWPFetchError(f"Open-Meteo GFS API request failed: {exc}") from exc

    # ---- Timezone returned by the API (bytes or str) -------------------------
    try:
        tz_name = response.Timezone()
        if isinstance(tz_name, bytes):
            tz_name = tz_name.decode("utf-8")
        tz_name = str(tz_name)
        pd.Timestamp.now(tz=tz_name)  # validate
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not use API timezone (%s); falling back to UTC.", exc)
        tz_name = "UTC"

    # ---- Current conditions ----------------------------------------------------
    current = response.Current()
    current_dict: dict[str, Any] = {
        "time_utc": datetime.fromtimestamp(current.Time(), tz=timezone.utc).isoformat(),
        **{var: round(float(current.Variables(i).Value()), 2) for i, var in enumerate(CURRENT_VARS)},
    }
    current_dict = sanitize(current_dict)
    desc, emoji = describe_weather_code(current_dict.get("weather_code"))
    current_dict["weather_description"] = desc
    current_dict["weather_emoji"] = emoji
    local_now = pd.Timestamp(current.Time(), unit="s", tz="UTC").tz_convert(tz_name)

    # ---- Hourly data (local time, starting from the current hour) --------------
    hourly = response.Hourly()
    hourly_time = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert(tz_name)

    hourly_data: dict[str, Any] = {"time": hourly_time}
    for i, var in enumerate(HOURLY_VARS):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    hourly_df = pd.DataFrame(hourly_data)

    start_idx = int(hourly_df["time"].searchsorted(local_now.floor("h")))
    start_idx = min(start_idx, max(len(hourly_df) - 1, 0))
    next_24h = hourly_df.iloc[start_idx:start_idx + 24].copy()

    records = next_24h.copy()
    records["time"] = records["time"].dt.strftime("%Y-%m-%dT%H:%M%z")
    hourly_list = records.round(2).to_dict(orient="records")

    # ---- Daily data (dates in the location's local timezone) -------------------
    daily = response.Daily()
    daily_time = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).tz_convert(tz_name)

    daily_data: dict[str, Any] = {"date": daily_time.strftime("%Y-%m-%d")}
    for i, var in enumerate(DAILY_VARS):
        daily_data[var] = daily.Variables(i).ValuesAsNumpy()
    daily_list = pd.DataFrame(daily_data).round(2).to_dict(orient="records")

    # ---- Air quality + alerts --------------------------------------------------
    air_quality = fetch_air_quality(location.lat, location.lon) if include_air_quality else {}
    alerts = _extract_current_alerts(next_24h, air_quality)

    result = NWPResult(
        location=location,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        timezone=tz_name,
        local_time=local_now.strftime("%Y-%m-%d %H:%M %Z"),
        current=current_dict,
        hourly_next_24h=hourly_list,
        daily_7day=daily_list,
        air_quality=air_quality,
        alerts=alerts,
    )
    logger.info("Fetched NWP data for %s (%s, %s) tz=%s", place_name, location.lat, location.lon, tz_name)
    return result


# ------------------------------------------------------------------------------
# Manual smoke test:  python wrf_simulator.py
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print(json.dumps(fetch_nwp_data("Shimla, Himachal Pradesh").to_dict(), indent=2, default=str))
