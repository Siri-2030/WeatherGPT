"""
smoke_test.py - quick end-to-end check of a RUNNING backend (local or Render).

    python smoke_test.py                                  # http://localhost:8000
    python smoke_test.py https://weathergpt-backend.onrender.com

Exit code 0 = everything works. Handy 5 minutes before a demo.
"""
import json
import sys
import time

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
failures = 0


def check(name, fn):
    global failures
    start = time.time()
    try:
        detail = fn()
        print(f"  PASS  {name:<28} {time.time() - start:5.1f}s  {detail}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  FAIL  {name:<28} {time.time() - start:5.1f}s  {exc}")


def health():
    r = requests.get(f"{BASE}/health", timeout=90)  # first call may wake a sleeping Render service
    r.raise_for_status()
    j = r.json()
    return f"llm={j['llm']}"


def weather():
    r = requests.post(f"{BASE}/weather", json={"location": "Vijayawada", "forecast_days": 3}, timeout=60)
    r.raise_for_status()
    j = r.json()
    json.dumps(j, allow_nan=False)
    cur = j["current"]
    return f"{j['location']['display_name']}: {cur['temperature_2m']}C, {cur['weather_description']}"


def weather_city_state():
    r = requests.post(f"{BASE}/weather", json={"location": "Shimla, Himachal Pradesh"}, timeout=60)
    r.raise_for_status()
    return r.json()["location"]["display_name"]


def chat():
    r = requests.post(f"{BASE}/chat", json={"message": "Will it rain in Chennai tomorrow?"}, timeout=120)
    r.raise_for_status()
    j = r.json()
    assert j["answer"] and not j["error"], j
    return f"{len(j['answer'])} chars in {j['latency_ms']} ms"


def chat_followup():
    history = [{"role": "user", "content": "Weather in Pune today?"},
               {"role": "assistant", "content": "Pune is partly cloudy, about 28 C."}]
    r = requests.post(f"{BASE}/chat", json={"message": "and tomorrow?", "history": history}, timeout=120)
    r.raise_for_status()
    j = r.json()
    assert "pune" in (j["location_used"] or "").lower(), f"follow-up lost context: {j['location_used']}"
    return f"location_used={j['location_used']}"


print(f"Smoke-testing {BASE}")
for label, func in [("GET /health", health), ("POST /weather", weather),
                    ("POST /weather (City, State)", weather_city_state),
                    ("POST /chat", chat), ("POST /chat (follow-up)", chat_followup)]:
    check(label, func)
print("\nAll good!" if not failures else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
