"""Free web search tool for the Reachy assistant.

Uses DuckDuckGo through the `ddgs` package: no API key, no account. Results are trimmed hard because the
local LLM is small and slow at reading prompts (about 10-15 tokens/s on the laptop CPU), so every extra
token of search output costs real latency.

Works with any OpenAI-style tool calling backend (llama-server, Ollama, the speech-to-speech server) and
can be wrapped for Gemini's FunctionDeclaration format with the same fields.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet. Use for anything that may have changed recently or that you are unsure of: "
            "news, sports scores, prices, release dates, opening hours, or facts about people, "
            "places and products. Do not use for weather (use get_weather), music, robot motion, "
            "or the user's calendar, tasks or email."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Short search query, a few keywords."},
                "kind": {"type": "string", "enum": ["web", "news"], "description": "'news' for current events."},
                "timelimit": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Only results from this period. Optional.",
                },
            },
            "required": ["query"],
        },
    },
}

_TIME = {"day": "d", "week": "w", "month": "m", "year": "y"}
_CACHE_TTL_S = 300.0
_MIN_GAP_S = 1.0          # be polite to DuckDuckGo; avoids rate-limit errors on rapid repeat calls
_MAX_SNIPPET = 150
_MAX_TITLE = 80
_DEFAULT_RESULTS = 3      # fewer results = fewer tokens for the model to read = faster spoken answer

_lock = threading.Lock()
_last_call = 0.0
_cache: dict[tuple, tuple[float, dict]] = {}


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def web_search(query: str, kind: str = "web", timelimit: str | None = None, max_results: int = _DEFAULT_RESULTS) -> dict[str, Any]:
    """Return {"results": [{"title", "snippet"[, "date"]}...]} or {"error": "..."}. Never raises.

    URLs and sources are deliberately left out: the model only speaks the answer and never reads links aloud.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "empty search query"}
    kind = kind if kind in ("web", "news") else "web"
    tl = _TIME.get(timelimit or "")
    max_results = max(1, min(int(max_results or _DEFAULT_RESULTS), 6))

    key = (query.lower(), kind, tl, max_results)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_S:
            return hit[1]

    try:
        from ddgs import DDGS
    except ImportError:
        return {"error": "web search is not installed (pip install ddgs)"}

    last_err = ""
    for attempt in range(2):
        with _lock:                                   # one search at a time, spaced out
            global _last_call
            wait = _MIN_GAP_S - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        try:
            ddgs = DDGS(timeout=8)
            if kind == "news":
                # Ask for extra, then keep the newest few: for "who won" style questions the latest article is
                # the one that matters, and the engine's own ranking often puts older previews first.
                raw = ddgs.news(query, max_results=max(max_results * 2, 6), timelimit=tl)
                rows = [{"title": r.get("title"), "snippet": r.get("body"), "url": r.get("url"),
                         "date": (r.get("date") or "")[:10], "source": r.get("source")} for r in raw]
                rows.sort(key=lambda r: r["date"], reverse=True)
                rows = rows[:max_results]
            else:
                raw = ddgs.text(query, max_results=max_results, timelimit=tl)
                rows = [{"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")} for r in raw]
            out_rows = []
            for r in rows:
                item = {"title": _clip(r["title"], _MAX_TITLE), "snippet": _clip(r["snippet"], _MAX_SNIPPET)}
                if r.get("date"):
                    item["date"] = r["date"]
                out_rows.append(item)
            result = {"query": query, "count": len(out_rows), "results": out_rows}
            if not out_rows:
                result["note"] = "no results; try different keywords"
            with _lock:
                _cache[key] = (time.monotonic(), result)
            return result
        except Exception as e:                        # network down, rate limit, parse error
            last_err = f"{type(e).__name__}: {e}"[:160]
            log.warning("web_search attempt %d failed: %s", attempt + 1, last_err)
            time.sleep(1.5)
    return {"error": f"search failed ({last_err}). Tell the user you could not reach the internet."}


if __name__ == "__main__":      # quick manual check:  python web_search.py "query"
    import json
    import sys

    print(json.dumps(web_search(" ".join(sys.argv[1:]) or "Reachy Mini robot"), indent=1))
