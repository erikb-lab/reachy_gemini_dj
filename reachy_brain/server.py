"""WebSocket server. Run: python -m reachy_brain.server --model path/to/model.gguf [--mmproj mmproj.gguf]

Protocol (one connection per robot):
  robot -> brain: binary = 16 kHz mono int16 mic PCM; JSON {"type": "hello", "tools": [...], ...},
                  {"type": "tool_result", "id", "result"}, {"type": "frame", "id", "jpeg_b64"},
                  {"type": "barge_in"}, {"type": "new_conversation"}, {"type": "ping"}
  brain -> robot: binary = 24 kHz mono int16 speech PCM; JSON {"type": "ready"|"state"|"heard"|"said"|"clear"|
                  "tool_call"|"request_frame"|"event"|"error"|"turn_done"|"pong", ...}
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

import websockets

from .config import BrainConfig
from .llm import LlamaServer
from .session import BrainSession
from .stt import STT
from .tts import make_tts

log = logging.getLogger("reachy_brain")


class Brain:
    def __init__(self, cfg: BrainConfig, llm=None, stt=None, tts=None):
        self.cfg = cfg
        self.llm = llm or LlamaServer(cfg)
        self.stt, self.tts = stt, tts
        self.active: BrainSession | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        log.info("loading models...")
        if isinstance(self.llm, LlamaServer):
            self.llm.start()
        self.stt = self.stt or STT(self.cfg.stt_model, self.cfg.stt_threads)
        self.tts = self.tts or make_tts(self.cfg.tts_engine, self.cfg.tts_voice, self.cfg.tts_threads)
        log.info("brain ready on ws://%s:%d", self.cfg.host, self.cfg.port)

    async def handler(self, ws) -> None:
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()

        def _send(data) -> None:
            """Safe from any thread. On the loop thread it schedules; elsewhere it waits so audio stays ordered."""
            if threading.get_ident() == loop_thread:
                loop.create_task(ws.send(data))
                return
            try:
                asyncio.run_coroutine_threadsafe(ws.send(data), loop).result(10)
            except Exception:
                pass

        sess = BrainSession(self.cfg, self.llm, self.stt, self.tts,
                            lambda obj: _send(json.dumps(obj)), _send)
        with self._lock:
            if self.active:                   # one robot at a time: newest connection wins
                self.active.close()
            self.active = sess
        log.info("robot connected from %s", ws.remote_address)
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    sess.on_audio(msg)        # VAD is cheap enough to run on the loop
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "hello":
                    sess.on_hello(m)
                elif t == "tool_result":
                    sess.on_tool_result(m)
                elif t == "frame":
                    sess.on_frame(m)
                elif t == "barge_in":
                    sess.barge_in()
                elif t == "new_conversation":
                    sess.new_conversation()
                elif t == "ping":
                    _send(json.dumps({"type": "pong"}))
        except websockets.ConnectionClosed:
            pass
        finally:
            sess.close()
            with self._lock:
                if self.active is sess:
                    self.active = None
            log.info("robot disconnected")

    async def serve(self) -> None:
        async with websockets.serve(self.handler, self.cfg.host, self.cfg.port, max_size=8 * 1024 * 1024,
                                    ping_interval=20, ping_timeout=20):
            await asyncio.Future()


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = BrainConfig.from_args(argv)
    brain = Brain(cfg)
    brain.load()
    try:
        asyncio.run(brain.serve())
    finally:
        brain.llm.stop()


if __name__ == "__main__":
    main()
