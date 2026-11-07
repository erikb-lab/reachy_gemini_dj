"""One-shot Gemini text answers for the local brain's "ask Gemini" tool. Runs on the robot, so the Gemini API key
never leaves it: the brain asks the robot (a tool call), the robot asks Google and returns only the answer."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"


def ask_gemini(question: str, api_key: str, model: str = MODEL) -> dict[str, Any]:
    """Never raises."""
    question = (question or "").strip()[:4000]
    if not api_key:
        return {"error": "No Gemini API key configured."}
    if not question:
        return {"error": "No question given."}
    try:
        from google import genai
        from google.genai import types
        r = genai.Client(api_key=api_key).models.generate_content(
            model=model, contents=question,
            config=types.GenerateContentConfig(
                system_instruction="Answer for a voice assistant: plain spoken sentences, no markdown, no lists, "
                                   "at most four sentences.",
                max_output_tokens=400))
        text = (r.text or "").strip()
        return {"answer": text} if text else {"error": "Gemini returned no answer."}
    except Exception as e:  # network, quota, bad key...
        log.warning("ask_gemini failed: %s", type(e).__name__)
        return {"error": f"Gemini is unavailable ({type(e).__name__})."}
