"""Brain server configuration (CLI flags override environment variables, which override defaults)."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class BrainConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    # LLM: either point at a running llama-server (llama_url) or let the brain start one.
    llama_url: str = ""
    llama_server_bin: str = "llama-server"
    model_path: str = ""
    mmproj_path: str = ""
    llm_port: int = 8089
    llm_threads: int = 4
    llm_ctx: int = 6144             # per conversation; llama-server gets llm_ctx * 2 (chat slot + vision slot)
    history_chars: int = 9000       # trim old turns (in one go, down to 60%) past this many characters
    slot_cache_dir: str = "~/.cache/reachy_brain/slots"   # saved prompt caches (llama-server --slot-save-path); never /tmp
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.8
    # STT / TTS
    stt_model: str = "base.en"
    stt_threads: int = 3
    tts_engine: str = "pocket"
    tts_voice: str = "alba"
    tts_threads: int = 3
    # Turn taking
    vad_aggressiveness: int = 2
    end_silence_ms: int = 650
    max_utterance_s: float = 20.0
    tool_timeout_s: float = 45.0
    tool_result_chars: int = 1500
    # Gemini help for hard questions (the key can also arrive from the robot app in its hello message)
    gemini_api_key: str = ""
    assist_model: str = "gemini-2.5-flash"
    persona: str = (
        "You are Reachy Mini, a witty tabletop robot DJ and companion. You speak aloud, so answer in one or two "
        "short plain sentences, with no emojis, markdown or lists, and never say tool names. Use your tools for "
        "anything about music, motion, the camera, weather, the web, FRC, timers or memory, and only state facts "
        "that appear in tool results."
    )

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "BrainConfig":
        d = cls()
        p = argparse.ArgumentParser(description="Reachy brain server")
        p.add_argument("--host", default=_env("BRAIN_HOST", d.host))
        p.add_argument("--port", type=int, default=int(_env("BRAIN_PORT", str(d.port))))
        p.add_argument("--llama-url", default=_env("LLAMA_URL"), help="use an already running llama-server")
        p.add_argument("--llama-server-bin", default=_env("LLAMA_SERVER_BIN", d.llama_server_bin))
        p.add_argument("--model", dest="model_path", default=_env("BRAIN_MODEL"))
        p.add_argument("--mmproj", dest="mmproj_path", default=_env("BRAIN_MMPROJ"))
        p.add_argument("--llm-threads", type=int, default=int(_env("BRAIN_LLM_THREADS", str(d.llm_threads))))
        p.add_argument("--stt-threads", type=int, default=int(_env("BRAIN_STT_THREADS", str(d.stt_threads))))
        p.add_argument("--slot-cache-dir", dest="slot_cache_dir", default=_env("BRAIN_SLOT_CACHE_DIR", d.slot_cache_dir),
                       help="where llama-server saves warmed prompt caches (on disk, not /tmp)")
        p.add_argument("--tts-voice", default=_env("BRAIN_TTS_VOICE", d.tts_voice))
        p.add_argument("--gemini-key", dest="gemini_api_key",
                       default=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
        a = p.parse_args(argv)
        for k, v in vars(a).items():
            setattr(d, k, v)
        return d
