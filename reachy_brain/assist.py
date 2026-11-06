"""Server-side tools that are not robot actions: web search, weather, camera look, and Gemini for hard questions."""
from __future__ import annotations

import logging
from typing import Any, Callable

from .weather import SCHEMA as WEATHER_SCHEMA, get_weather
from .web_search import SCHEMA as SEARCH_SCHEMA, web_search

log = logging.getLogger(__name__)

LOOK_SCHEMA = {"type": "function", "function": {
    "name": "look",
    "description": "Use the robot's camera to see. Call this when the user asks what you see, to look at something, "
                   "or to read or identify something in front of you.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "What to look for or answer about the image."}}}}}

ASK_GEMINI_SCHEMA = {"type": "function", "function": {
    "name": "ask_gemini",
    "description": "Ask a stronger cloud AI. Use only for hard requests: multi-step reasoning, math, code, long or "
                   "creative writing, or detailed explanations you are not confident about. Not for simple chat, "
                   "music, motion, weather or lookups.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "The full question, with any needed context."}},
        "required": ["question"]}}}

# Tools whose result must be read back to the user (an "info" tool): the model answers from the result.
SERVER_TOOLS: dict[str, dict] = {
    "web_search": SEARCH_SCHEMA, "get_weather": WEATHER_SCHEMA, "look": LOOK_SCHEMA, "ask_gemini": ASK_GEMINI_SCHEMA,
}


def ask_gemini(question: str, api_key: str, model: str) -> dict[str, Any]:
    """One-shot Gemini text call. Never raises."""
    if not api_key:
        return {"error": "No Gemini API key configured."}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        r = client.models.generate_content(
            model=model, contents=question,
            config=types.GenerateContentConfig(
                system_instruction="Answer for a voice assistant: plain spoken sentences, no markdown, no lists, "
                                   "at most four sentences.",
                max_output_tokens=400))
        text = (r.text or "").strip()
        return {"answer": text} if text else {"error": "Gemini returned no answer."}
    except Exception as e:  # network, quota, bad key...
        log.warning("ask_gemini failed: %s", e)
        return {"error": f"Gemini is unavailable ({type(e).__name__})."}


def run_server_tool(name: str, args: dict, *, gemini_key: str, assist_model: str,
                    look: Callable[[str], dict]) -> dict[str, Any]:
    if name == "web_search":
        return web_search(**args)
    if name == "get_weather":
        return get_weather(**args)
    if name == "look":
        return look(str(args.get("question", "")))
    if name == "ask_gemini":
        return ask_gemini(str(args.get("question", "")), gemini_key, assist_model)
    return {"error": f"unknown server tool {name}"}
