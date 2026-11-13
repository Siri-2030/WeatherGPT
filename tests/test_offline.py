"""
Offline tests for WeatherGPT -- no network, no API keys needed.

Run from the project root:
    python -m unittest discover -s tests -v        (or: pytest -v)
"""
import json
import math
import os
import sys
import types
import unittest

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# If the real libraries are not installed (e.g. a bare CI box) stub the three
# HTTP helpers so the pure-logic code can still be imported and tested.
for _name in ("requests_cache", "retry_requests", "openmeteo_requests"):
    try:
        __import__(_name)
    except ImportError:
        mod = types.ModuleType(_name)
        if _name == "requests_cache":
            mod.CachedSession = lambda *a, **k: requests.Session()
        elif _name == "retry_requests":
            mod.retry = lambda session, **k: session
        else:
            class _Client:  # noqa: D401
                def __init__(self, session=None): ...
                def weather_api(self, url, params=None): raise RuntimeError("stub")
            mod.Client = _Client
        sys.modules[_name] = mod

import prompts  # noqa: E402
import voice_engine  # noqa: E402
import wrf_simulator as ws  # noqa: E402
import llm_router  # noqa: E402


# --------------------------------------------------------------------------- fakes
class _Var:
    def __init__(self, values):
        self._v = np.array(values, dtype=float)

    def Value(self):
        return float(self._v[0])

    def ValuesAsNumpy(self):
        return self._v


class _Block:
    def __init__(self, start, end, interval, variables):
        self._s, self._e, self._i, self._vars = start, end, interval, variables

    def Time(self): return self._s
    def TimeEnd(self): return self._e
    def Interval(self): return self._i
    def Variables(self, i): return self._vars[i]


class FakeResponse:
    """Mimics an openmeteo_sdk response for Asia/Kolkata (UTC+5:30), 2 days."""
    # 2026-11-13 00:00 IST == 2026-11-12 18:30 UTC
    DAY0_UTC = int(pd.Timestamp("2026-11-12 18:30", tz="UTC").timestamp())

    def Timezone(self): return b"Asia/Kolkata"

    def Current(self):
        # "now" = 2026-11-13 10:15 IST  -> 04:45 UTC
        now = int(pd.Timestamp("2026-11-13 04:45", tz="UTC").timestamp())
        vals = [_Var([31.2]), _Var([60]), _Var([0.0]), _Var([2]), _Var([12]), _Var([30]), _Var([800])]
        return _Block(now, now + 900, 900, vals)

    def Hourly(self):
        n = 48
        vals = []
        for name in ws.HOURLY_VARS:
            data = list(range(n))
            if name == "temperature_2m":
                data = [20 + (i % 12) for i in range(n)]   # max 31 C -> no heat alert
            if name == "soil_moisture_0_to_1cm":
                data = [float("nan")] * n          # <- the NaN that used to give HTTP 500
            if name == "wind_gusts_10m":
                data = [10.0] * n
            if name == "cape":
                data = [100.0] * n
            if name == "precipitation":
                data = [0.0] * n
            vals.append(_Var(data))
        return _Block(self.DAY0_UTC, self.DAY0_UTC + n * 3600, 3600, vals)

    def Daily(self):
        vals = [_Var([30 + i, 31 + i]) for i, _ in enumerate(ws.DAILY_VARS)]
        vals[3] = _Var([float("nan"), 40])          # NaN in daily data too
        return _Block(self.DAY0_UTC, self.DAY0_UTC + 2 * 86400, 86400, vals)


# --------------------------------------------------------------------------- tests
class SanitizeTests(unittest.TestCase):
    def test_nan_inf_become_none_and_json_is_strict(self):
        data = {"a": float("nan"), "b": [1.5, float("inf")], "c": np.float64("nan"), "d": np.int64(3)}
        clean = ws.sanitize(data)
        self.assertEqual(clean, {"a": None, "b": [1.5, None], "c": None, "d": 3})
        json.dumps(clean, allow_nan=False)  # must not raise (this is what FastAPI does)


class LookupTests(unittest.TestCase):
    def test_weather_code_and_aqi(self):
        self.assertEqual(ws.describe_weather_code(95)[0], "Thunderstorm")
        self.assertEqual(ws.describe_weather_code(None)[0], "Unknown")
        self.assertEqual(ws.describe_weather_code(1234)[0], "Unknown")
        self.assertEqual(ws.aqi_category(42), "Good")
        self.assertEqual(ws.aqi_category(160), "Unhealthy")
        self.assertEqual(ws.aqi_category(None), "Unknown")


class GeocodingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._orig = ws._search_places
        shimla_wrong = {"name": "Shimla", "latitude": 1.0, "longitude": 2.0, "admin1": "Elsewhere", "country": "Nowhere"}
        shimla_right = {"name": "Shimla", "latitude": 31.1048, "longitude": 77.1734, "admin1": "Himachal Pradesh", "country": "India"}

        def fake(name, count=10):
            self.calls.append(name)
            if name.lower() == "shimla":
                return [shimla_wrong, shimla_right]
            return []                                  # "Shimla, Himachal Pradesh" -> nothing (real API behaviour)
        ws._search_places = fake
        ws._LOCATION_CACHE.pop("shimla, himachal pradesh", None)

    def tearDown(self):
        ws._search_places = self._orig

    def test_city_state_query_uses_hints(self):
        loc = ws.geocode_location("Shimla, Himachal Pradesh")
        self.assertEqual(self.calls, ["Shimla"])
        self.assertAlmostEqual(loc.lat, 31.1048, places=3)
        self.assertEqual(loc.display_name, "Shimla, Himachal Pradesh, India")

    def test_result_is_cached(self):
        ws.geocode_location("Shimla, Himachal Pradesh")
        ws.geocode_location("shimla,   himachal pradesh")
        self.assertEqual(len(self.calls), 1)

    def test_seeded_city_needs_no_api_call(self):
        loc = ws.geocode_location("Vijayawada")
        self.assertEqual(self.calls, [])
        self.assertEqual(loc.country, "India")

    def test_empty_name(self):
        with self.assertRaises(ws.GeocodingError):
            ws.geocode_location("   ")

    def test_unknown_place(self):
        with self.assertRaises(ws.GeocodingError):
            ws.geocode_location("Qwertyuiop")


class AlertTests(unittest.TestCase):
    def test_alert_rules(self):
        df = pd.DataFrame({
            "wind_gusts_10m": [95.0] + [10.0] * 23, "cape": [3000.0] * 24,
            "precipitation": [5.0] * 24, "temperature_2m": [46.0] * 24,
        })
        alerts = ws._extract_current_alerts(df, {"us_aqi": 210})
        types_ = {a["type"]: a["severity"] for a in alerts}
        self.assertEqual(types_["High Wind"], "Severe")
        self.assertEqual(types_["Thunderstorm / Severe Convection Risk"], "Severe")
        self.assertEqual(types_["Heavy Rainfall / Flood Risk"], "Severe")  # 120 mm
        self.assertEqual(types_["Heat Wave"], "Severe")
        self.assertEqual(types_["Poor Air Quality"], "Severe")

    def test_calm_day_has_no_alerts(self):
        df = pd.DataFrame({"wind_gusts_10m": [10.0] * 24, "cape": [10.0] * 24,
                           "precipitation": [0.0] * 24, "temperature_2m": [28.0] * 24})
        self.assertEqual(ws._extract_current_alerts(df, {"us_aqi": 40}), [])


class FetchNWPTests(unittest.TestCase):
    def setUp(self):
        self._api, self._aq = ws._openmeteo.weather_api, ws.fetch_air_quality
        ws._openmeteo.weather_api = lambda url, params=None: [FakeResponse()]
        ws.fetch_air_quality = lambda lat, lon: {"us_aqi": 88.0, "category": "Moderate"}

    def tearDown(self):
        ws._openmeteo.weather_api, ws.fetch_air_quality = self._api, self._aq

    def test_end_to_end_with_nan_values(self):
        out = ws.fetch_nwp_data("Vijayawada", forecast_days=2).to_dict()
        json.dumps(out, allow_nan=False)                       # JSON-safe
        self.assertEqual(out["timezone"], "Asia/Kolkata")
        self.assertTrue(out["local_time"].startswith("2026-11-13 10:15"))
        self.assertEqual(out["current"]["weather_description"], "Partly cloudy")
        # hourly now starts at the CURRENT local hour (10:00 IST), in local time
        self.assertEqual(len(out["hourly_next_24h"]), 24)
        self.assertTrue(out["hourly_next_24h"][0]["time"].startswith("2026-11-13T10:00+0530"))
        self.assertIsNone(out["hourly_next_24h"][0]["soil_moisture_0_to_1cm"])
        # daily dates are LOCAL dates (not the previous UTC day)
        self.assertEqual([d["date"] for d in out["daily_7day"]], ["2026-11-13", "2026-11-14"])
        self.assertIsNone(out["daily_7day"][0]["precipitation_probability_max"])
        self.assertEqual(out["air_quality"]["category"], "Moderate")
        self.assertEqual(out["alerts"], [])

    def test_api_failure_becomes_nwpfetcherror(self):
        def boom(url, params=None): raise RuntimeError("down")
        ws._openmeteo.weather_api = boom
        with self.assertRaises(ws.NWPFetchError):
            ws.fetch_nwp_data("Vijayawada")


class RouterHelperTests(unittest.TestCase):
    def test_extract_json_object_variants(self):
        self.assertEqual(llm_router._extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(llm_router._extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm_router._extract_json_object('Sure! {"a": 1} hope it helps'), {"a": 1})
        with self.assertRaises(ValueError):
            llm_router._extract_json_object("no json here")

    def test_content_to_text(self):
        self.assertEqual(llm_router._content_to_text("hi"), "hi")
        self.assertEqual(llm_router._content_to_text([{"type": "text", "text": "a"}, "b"]), "ab")
        self.assertEqual(llm_router._content_to_text("<think>hmm</think>Answer"), "Answer")

    def test_history_formatting(self):
        text = llm_router._format_history([{"role": "user", "content": "Weather in Pune?"},
                                          {"role": "system", "content": "ignored"}])
        self.assertEqual(text, "user: Weather in Pune?")
        self.assertEqual(llm_router._format_history(None), "(none)")

    def test_classify_intent_with_fake_llm(self):
        class Msg:  # minimal LangChain-like message
            def __init__(self, c): self.content = c
        class FakeLLM:
            def invoke(self, _): return Msg('```json\n{"location": "null", "forecast_days": 1, "needs_weather_data": true}\n```')
        orig = llm_router.get_llm
        llm_router.get_llm = lambda: FakeLLM()
        try:
            out = llm_router.classify_intent("will it rain tomorrow?")
        finally:
            llm_router.get_llm = orig
        self.assertIsNone(out["location"])          # "null" string -> None
        self.assertEqual(out["forecast_days"], 2)   # "tomorrow" forces >= 2


class PromptTests(unittest.TestCase):
    def test_history_goes_between_system_and_final_user_message(self):
        msgs = prompts.build_final_prompt(
            "and tomorrow?", {"x": 1}, "general",
            history=[{"role": "user", "content": "Weather in Pune?"},
                     {"role": "assistant", "content": "Sunny, 30C."}])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertIn('"x": 1', msgs[-1]["content"])
        self.assertIn("and tomorrow?", msgs[-1]["content"])

    def test_unknown_persona_falls_back(self):
        self.assertEqual(prompts.build_final_prompt("q", {}, "nonsense")[0]["role"], "system")

    def test_persona_detection(self):
        self.assertEqual(prompts.detect_persona("Should I irrigate my crop?"), "agriculture")
        self.assertEqual(prompts.detect_persona("cyclone warning"), "disaster")


class VoiceHelperTests(unittest.TestCase):
    def test_clean_for_speech(self):
        raw = "**Alert:** heavy rain 🌧️\n- Stay indoors\n[IMD](http://x.y) says `ok`"
        clean = voice_engine._clean_for_speech(raw)
        for bad in ("*", "`", "🌧", "http", "- "):
            self.assertNotIn(bad, clean)
        self.assertIn("Stay indoors", clean)
        self.assertIn("IMD says ok", clean)

    def test_audio_mime(self):
        self.assertEqual(voice_engine.audio_mime(b"RIFF....WAVE"), "audio/wav")
        self.assertEqual(voice_engine.audio_mime(b"ID3\x03"), "audio/mp3")

    def test_empty_text_gives_error_not_crash(self):
        audio, err = voice_engine.generate_voice_response("   ")
        self.assertIsNone(audio)
        self.assertTrue(err)


@unittest.skipUnless(
    all(__import__("importlib").util.find_spec(m) for m in ("fastapi", "httpx")),
    "fastapi/httpx not installed",
)
class ApiTests(unittest.TestCase):
    def test_endpoints(self):
        from fastapi.testclient import TestClient
        import main
        client = TestClient(main.app)
        self.assertEqual(client.get("/health").json()["status"], "ok")
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.post("/chat", json={"message": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
