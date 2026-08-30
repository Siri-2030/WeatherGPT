
"""
wrf_simulator.py
================================================================================
Data Ingestor Module for WeatherGPT.

NOTE ON NAMING: This module is named `wrf_simulator` to match the requested
architecture, but it does NOT run an actual WRF (Weather Research and
Forecasting) binary simulation -- that requires a compiled HPC model that
cannot run on free-tier hosting (Hugging Face Spaces / Render have no
Fortran/MPI compute nodes, no persistent disk for GRIB2 archives, and runs
take 10-60+ minutes on real HPC clusters).

Instead, this module treats the Open-Meteo API as a hosted, pre-computed
NWP proxy. Open-Meteo re-grids and serves the SAME underlying model outputs
(NOAA GFS, and regionally NOAA HRRR / DWD ICON / ECMWF) that a self-hosted
WRF run would be initialized from -- for free, with no API key, at
sub-second latency. This gives WeatherGPT real numerical model data without
needing to compile or host WRF ourselves.

If you later get access to real HPC resources, you only need to replace the
`fetch_nwp_data()` function's internals with a call to your WPS/WRF output
NetCDF parser -- the rest of the pipeline (geocoding, structuring, LLM
injection) stays identical. See the `# HPC-UPGRADE-PATH` comments below.

Responsibilities:
1. Geocode a free-text location string -> (lat, lon) using free OSM Nominatim.
2. Fetch current + forecast NWP variables via the Open-Meteo GFS API.
3. Return a clean, structured Python dict ready for LLM prompt injection.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import openmeteo_requests
import pandas as pd
import requests_cache
#from geopy.geocoders import Nominatim
#from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from retry_requests import retry
import requests


logger = logging.getLogger("weathergpt.wrf_simulator")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------
# Setup: cached + retrying HTTP session
# ------------------------------------------------------------------------------
_cache_session = requests_cache.CachedSession(
    ".weathergpt_cache",
    expire_after=900,
)

_retry_session = retry(
    _cache_session,
    retries=5,
    backoff_factor=0.2,
)

_openmeteo = openmeteo_requests.Client(
    session=_retry_session
)


# --------------------------------------------------------------------------
# Global geocoder
# --------------------------------------------------------------------------
# Open-Meteo Geocoding API provides worldwide location search.
# It avoids relying on the public Nominatim service, which can return
# HTTP 429 rate-limit errors on cloud-hosted applications.

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/gfs"
)

OPEN_METEO_AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ------------------------------------------------------------------------------
# NWP variables
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
    #"sunrise",
    #"sunset",
]


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
    current: dict[str, Any] = field(default_factory=dict)
    hourly_next_24h: list[dict[str, Any]] = field(default_factory=list)
    daily_7day: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, str]] = field(default_factory=list)

    source: str = (
        "Open-Meteo GFS API (NOAA GFS 0.25deg proxy)"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": {
                "query": self.location.query,
                "lat": self.location.lat,
                "lon": self.location.lon,
                "display_name": self.location.display_name,
                "country": self.location.country,
                "state": self.location.state,
            },
            "fetched_at_utc": self.fetched_at_utc,
            "current": self.current,
            "hourly_next_24h": self.hourly_next_24h,
            "daily_7day": self.daily_7day,
            "alerts": self.alerts,
            "source": self.source,
        }


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
# ------------------------------------------------------------------------------
# Geocoding
# ------------------------------------------------------------------------------
# Built-in coordinates for frequently used locations.
# This avoids unnecessary calls to the public Nominatim service.
_LOCATION_CACHE = {
    "vijayawada": GeoLocation(
        query="Vijayawada",
        lat=16.5062,
        lon=80.6480,
        display_name="Vijayawada, Andhra Pradesh, India",
        country="India",
        state="Andhra Pradesh",
    ),
    "vijayawada, andhra pradesh": GeoLocation(
        query="Vijayawada, Andhra Pradesh",
        lat=16.5062,
        lon=80.6480,
        display_name="Vijayawada, Andhra Pradesh, India",
        country="India",
        state="Andhra Pradesh",
    ),
}


def geocode_location(place_name: str) -> GeoLocation:
    """
    Convert a free-text place name into coordinates using the
    Open-Meteo Geocoding API.

    Supports locations worldwide and avoids relying on the
    public OpenStreetMap Nominatim service.
    """

    normalized = " ".join(
        place_name.strip().lower().split()
    )

    if not normalized:
        raise GeocodingError("Location name cannot be empty.")

    # --------------------------------------------------------------------------
    # LOCAL LOCATION CACHE
    # --------------------------------------------------------------------------

    if normalized in _LOCATION_CACHE:

        cached = _LOCATION_CACHE[normalized]

        logger.info(
            "Using cached coordinates for location: %s",
            place_name,
        )

        return GeoLocation(
            query=place_name,
            lat=cached.lat,
            lon=cached.lon,
            display_name=cached.display_name,
            country=cached.country,
            state=cached.state,
        )

    # --------------------------------------------------------------------------
    # OPEN-METEO GEOCODING
    # --------------------------------------------------------------------------

    try:

        response = requests.get(
            GEOCODING_URL,
            params={
                "name": place_name,
                "count": 5,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:

        raise GeocodingError(
            f"Geocoding service unavailable: {exc}"
        ) from exc

    # --------------------------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------------------------

    results = data.get("results") or []

    if not results:

        raise GeocodingError(
            f"Could not resolve location: '{place_name}'"
        )

    # Use the first Open-Meteo result.
    location = results[0]

    latitude = location.get("latitude")
    longitude = location.get("longitude")

    if latitude is None or longitude is None:

        raise GeocodingError(
            f"Geocoding returned incomplete coordinates for '{place_name}'"
        )

    # --------------------------------------------------------------------------
    # BUILD DISPLAY NAME
    # --------------------------------------------------------------------------

    name = location.get("name") or place_name
    admin1 = location.get("admin1")
    country = location.get("country")

    display_parts = [name]

    if admin1 and admin1.lower() != name.lower():
        display_parts.append(admin1)

    if country:
        display_parts.append(country)

    display_name = ", ".join(display_parts)

    result = GeoLocation(
        query=place_name,
        lat=round(float(latitude), 4),
        lon=round(float(longitude), 4),
        display_name=display_name,
        country=country,
        state=admin1,
    )

    # --------------------------------------------------------------------------
    # CACHE SUCCESSFUL RESULT
    # --------------------------------------------------------------------------

    _LOCATION_CACHE[normalized] = result

    logger.info(
        "Geocoded '%s' -> %s (%s, %s)",
        place_name,
        display_name,
        result.lat,
        result.lon,
    )

    return result

# ------------------------------------------------------------------------------
# Alert extraction
# ------------------------------------------------------------------------------
def _extract_current_alerts(
    hourly_df: pd.DataFrame,
) -> list[dict[str, str]]:
    """
    Lightweight rule-based severe-weather flagging derived directly
    from NWP numerical output.

    These are simplified hackathon-demo thresholds and are NOT official
    meteorological warnings.
    """

    alerts: list[dict[str, str]] = []

    if hourly_df.empty:
        return alerts

    next_24h = hourly_df.iloc[:24]

    # --------------------------------------------------------------------------
    # High wind
    # --------------------------------------------------------------------------
    max_gust = next_24h["wind_gusts_10m"].max()

    if pd.notna(max_gust) and max_gust >= 60:

        alerts.append(
            {
                "type": "High Wind",
                "severity": (
                    "Severe"
                    if max_gust >= 90
                    else "Moderate"
                ),
                "message": (
                    f"Wind gusts up to {max_gust:.0f} km/h "
                    "expected in the next 24h."
                ),
            }
        )

    # --------------------------------------------------------------------------
    # CAPE / thunderstorm risk
    # --------------------------------------------------------------------------
    max_cape = next_24h["cape"].max()

    if pd.notna(max_cape) and max_cape >= 1500:

        alerts.append(
            {
                "type": "Thunderstorm / Severe Convection Risk",
                "severity": (
                    "Severe"
                    if max_cape >= 2500
                    else "Moderate"
                ),
                "message": (
                    f"High CAPE ({max_cape:.0f} J/kg) indicates "
                    "strong potential for thunderstorms/hail."
                ),
            }
        )

    # --------------------------------------------------------------------------
    # Heavy rainfall
    # --------------------------------------------------------------------------
    max_precip = next_24h["precipitation"].sum()

    if pd.notna(max_precip) and max_precip >= 60:

        alerts.append(
            {
                "type": "Heavy Rainfall / Flood Risk",
                "severity": (
                    "Severe"
                    if max_precip >= 115
                    else "Moderate"
                ),
                "message": (
                    f"Cumulative rainfall of {max_precip:.0f} mm "
                    "expected in next 24h."
                ),
            }
        )

    return alerts


# ------------------------------------------------------------------------------
# Main NWP data fetch
# ------------------------------------------------------------------------------
def fetch_nwp_data(
    place_name: str,
    forecast_days: int = 7,
) -> NWPResult:
    """
    Main entry point:
        1. Geocode location
        2. Fetch structured NWP data
        3. Build current/hourly/daily structures
        4. Generate rule-based alerts

    HPC-UPGRADE-PATH:
    To swap in real WRF output later:

        1. Trigger WRF run for the geocoded lat/lon.
        2. Parse wrfout_*.nc using xarray/netCDF4.
        3. Map WRF variables such as:
             T2
             RAINNC
             U10
             V10
             etc.
           into the SAME:
             current
             hourly_next_24h
             daily_7day
           structure.

    This allows llm_router.py and prompts.py to remain unchanged.
    """

    # --------------------------------------------------------------------------
    # Geocode
    # --------------------------------------------------------------------------
    location = geocode_location(place_name)

    # --------------------------------------------------------------------------
    # API parameters
    # --------------------------------------------------------------------------
    params = {
        "latitude": location.lat,
        "longitude": location.lon,

        "hourly": HOURLY_VARS,

        "daily": DAILY_VARS,

        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "wind_gusts_10m",
            "cape",
        ],

        "forecast_days": min(
            max(forecast_days, 1),
            16,
        ),

        # IMPORTANT:
        # Open-Meteo calculates daily values in the location's
        # local timezone.
        "timezone": "auto",
    }

    # --------------------------------------------------------------------------
    # Fetch Open-Meteo data
    # --------------------------------------------------------------------------
    try:

        responses = _openmeteo.weather_api(
            OPEN_METEO_FORECAST_URL,
            params=params,
        )

        response = responses[0]

    except Exception as exc:  # noqa: BLE001

        raise NWPFetchError(
            f"Open-Meteo GFS API request failed: {exc}"
        ) from exc


    # ==========================================================================
    # CURRENT CONDITIONS
    # ==========================================================================
    current = response.Current()

    current_vars = params["current"]

    current_dict = {
        "time_utc": datetime.fromtimestamp(
            current.Time(),
            tz=timezone.utc,
        ).isoformat(),

        **{
            var: round(
                float(
                    current.Variables(i).Value()
                ),
                2,
            )
            for i, var in enumerate(current_vars)
        },
    }


    # ==========================================================================
    # HOURLY DATA
    # ==========================================================================
    hourly = response.Hourly()

    hourly_time = pd.date_range(
        start=pd.to_datetime(
            hourly.Time(),
            unit="s",
            utc=True,
        ),

        end=pd.to_datetime(
            hourly.TimeEnd(),
            unit="s",
            utc=True,
        ),

        freq=pd.Timedelta(
            seconds=hourly.Interval()
        ),

        inclusive="left",
    )

    hourly_data = {
        "time": hourly_time
    }

    for i, var in enumerate(HOURLY_VARS):

        hourly_data[var] = (
            hourly
            .Variables(i)
            .ValuesAsNumpy()
        )

    hourly_df = pd.DataFrame(
        hourly_data
    )

    # Keep first 24 hours
    hourly_records = (
        hourly_df
        .iloc[:24]
        .copy()
    )

    hourly_records["time"] = (
        hourly_records["time"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M%z"
        )
    )

    hourly_list = (
        hourly_records
        .round(2)
        .to_dict(
            orient="records"
        )
    )


    # ==========================================================================
    # DAILY DATA
    # ==========================================================================

    # --------------------------------------------------------------------------
    # OLD DAILY DATE HANDLING
    #
    # KEPT FOR REFERENCE AS REQUESTED.
    #
    # The problem with this version is that the timestamps are interpreted
    # directly as UTC dates. For India (UTC+05:30), local midnight appears
    # as 18:30 on the previous UTC date.
    # --------------------------------------------------------------------------

    # daily = response.Daily()
    #
    # daily_time = pd.date_range(
    #     start=pd.to_datetime(
    #         daily.Time(),
    #         unit="s",
    #         utc=True,
    #     ),
    #
    #     end=pd.to_datetime(
    #         daily.TimeEnd(),
    #         unit="s",
    #         utc=True,
    #     ),
    #
    #     freq=pd.Timedelta(
    #         seconds=daily.Interval()
    #     ),
    #
    #     inclusive="left",
    # )
    #
    # daily_data = {
    #     "date": daily_time.strftime("%Y-%m-%d")
    # }
    #
    # for i, var in enumerate(DAILY_VARS):
    #
    #     vals = (
    #         daily
    #         .Variables(i)
    #         .ValuesAsNumpy()
    #     )
    #
    #     daily_data[var] = vals
    #
    # daily_df = pd.DataFrame(
    #     daily_data
    # )
    #
    # daily_list = (
    #     daily_df
    #     .round(2)
    #     .to_dict(
    #         orient="records"
    #     )
    # )


    # --------------------------------------------------------------------------
    # NEW DAILY DATE HANDLING
    #
    # Convert the Open-Meteo daily timestamps from UTC to the timezone
    # returned by Open-Meteo for the requested location.
    #
    # Example for Vijayawada:
    #
    # API UTC timestamp:
    #     2026-08-28 18:30 UTC
    #
    # Vijayawada local time:
    #     2026-08-29 00:00 IST
    #
    # Therefore the correct date is:
    #     2026-08-29
    #
    # --------------------------------------------------------------------------

    daily = response.Daily()

    # First construct the timestamps as UTC.
    daily_time_utc = pd.date_range(
        start=pd.to_datetime(
            daily.Time(),
            unit="s",
            utc=True,
        ),

        end=pd.to_datetime(
            daily.TimeEnd(),
            unit="s",
            utc=True,
        ),

        freq=pd.Timedelta(
            seconds=daily.Interval()
        ),

        inclusive="left",
    )


    # --------------------------------------------------------------------------
    # Determine the timezone returned by Open-Meteo.
    # --------------------------------------------------------------------------
    try:
        api_timezone = response.Timezone()

    # Open-Meteo may return the timezone as bytes, e.g.
    # b'Europe/London'. Decode it before using it with pandas.
        if isinstance(api_timezone, bytes):
            api_timezone = api_timezone.decode("utf-8")

        api_timezone = str(api_timezone)

        daily_time = daily_time_utc.tz_convert(
            api_timezone
    )

        logger.info(
            "Using API timezone: %s",
            api_timezone,
    )

    except Exception as exc:

         logger.warning(
            "Could not determine API timezone (%s). "
            "Using UTC as fallback.",
             exc,
    )

         daily_time = daily_time_utc.tz_convert(
            "UTC"
    )

    # --------------------------------------------------------------------------
    # Build daily records
    # --------------------------------------------------------------------------
    daily_data = {
        "date": daily_time.strftime(
            "%Y-%m-%d"
        )
    }


    for i, var in enumerate(DAILY_VARS):

        vals = (
            daily
            .Variables(i)
            .ValuesAsNumpy()
        )

        daily_data[var] = vals


    daily_df = pd.DataFrame(
        daily_data
    )


    daily_list = (
        daily_df
        .round(2)
        .to_dict(
            orient="records"
        )
    )


    # ==========================================================================
    # ALERTS
    # ==========================================================================
    alerts = _extract_current_alerts(
        hourly_df
    )


    # ==========================================================================
    # FINAL RESULT
    # ==========================================================================
    result = NWPResult(

        location=location,

        fetched_at_utc=(
            datetime
            .now(timezone.utc)
            .isoformat()
        ),

        current=current_dict,

        hourly_next_24h=hourly_list,

        daily_7day=daily_list,

        alerts=alerts,
    )


    logger.info(
        "Fetched NWP data for %s (%s, %s)",
        place_name,
        location.lat,
        location.lon,
    )

    return result


# ------------------------------------------------------------------------------
# Air quality
# ------------------------------------------------------------------------------
def fetch_air_quality(
    lat: float,
    lon: float,
) -> dict[str, Any]:
    """
    Bonus: free Open-Meteo Air Quality API.

    Useful for smart-city / urban-planning use cases.
    """

    params = {
        "latitude": lat,
        "longitude": lon,

        "current": [
            "pm2_5",
            "pm10",
            "us_aqi",
            "ozone",
            "carbon_monoxide",
        ],
    }

    try:

        responses = _openmeteo.weather_api(
            OPEN_METEO_AIR_QUALITY_URL,
            params=params,
        )

        current = responses[0].Current()

        keys = params["current"]

        return {
            k: round(
                float(
                    current
                    .Variables(i)
                    .Value()
                ),
                1,
            )

            for i, k in enumerate(keys)
        }

    except Exception as exc:  # noqa: BLE001

        logger.warning(
            "Air quality fetch failed: %s",
            exc,
        )

        return {}


# ------------------------------------------------------------------------------
# Manual smoke test
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    import json

    result = fetch_nwp_data(
        "Shimla, Himachal Pradesh"
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            default=str,
        )
    )

