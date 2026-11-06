"""Live weather tool for the Reachy assistant, via Open-Meteo (free, no API key, no account).

Search snippets never contain live conditions, so weather gets its own tool. The result is one short
line the model can read out, which also keeps the prompt small (the local LLM reads about 16 tokens/s).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather and today's high and low for a city. Use for any weather question.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Chicago' or 'Paris, France'."},
                "units": {"type": "string", "enum": ["fahrenheit", "celsius"], "description": "Default fahrenheit."},
            },
            "required": ["location"],
        },
    },
}

_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}
_CACHE_TTL_S = 600.0
_cache: dict[tuple, tuple[float, dict]] = {}


def _get(url: str, params: dict) -> dict:
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "reachy-gemini-dj"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.load(r)


def get_weather(location: str, units: str = "fahrenheit") -> dict[str, Any]:
    """Return {"summary": "..."} or {"error": "..."}. Never raises."""
    location = (location or "").strip()
    if not location:
        return {"error": "no location given; ask the user which city"}
    units = "celsius" if str(units).lower().startswith("c") else "fahrenheit"
    key = (location.lower(), units)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL_S:
        return hit[1]
    try:
        # The geocoder matches on the city name only, so split "Paris, France" and use the rest to pick.
        name, _, hint = location.partition(",")
        g = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": name.strip(), "count": 5, "language": "en"})
        places = g.get("results") or []
        if hint.strip():
            h = hint.strip().lower()
            places = [p for p in places if h in (p.get("country", "") + " " + p.get("admin1", "")).lower()] or places
        if not places:
            return {"error": f"could not find a place called {location}"}
        p = places[0]
        deg = "°F" if units == "fahrenheit" else "°C"
        w = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": 1,
            "temperature_unit": units, "wind_speed_unit": "mph" if units == "fahrenheit" else "kmh",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        })
        c, d = w["current"], w["daily"]
        place = ", ".join(x for x in (p.get("name"), p.get("admin1") if p.get("country_code") == "US" else p.get("country")) if x)
        wind_u = "mph" if units == "fahrenheit" else "km/h"
        s = (f"{place}: {round(c['temperature_2m'])}{deg} and {_WMO.get(c['weather_code'], 'unsettled')}, "
             f"feels like {round(c['apparent_temperature'])}{deg}, wind {round(c['wind_speed_10m'])} {wind_u}. "
             f"Today's high {round(d['temperature_2m_max'][0])}{deg}, low {round(d['temperature_2m_min'][0])}{deg}")
        pop = d.get("precipitation_probability_max", [None])[0]
        if pop is not None:
            s += f", {pop}% chance of precipitation"
        out = {"summary": s + "."}
        _cache[key] = (time.monotonic(), out)
        return out
    except Exception as e:
        log.warning("get_weather failed: %s", e)
        return {"error": f"could not reach the weather service ({type(e).__name__}). Tell the user."}


if __name__ == "__main__":
    import sys
    print(json.dumps(get_weather(" ".join(sys.argv[1:]) or "Chicago"), indent=1))
