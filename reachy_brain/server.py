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
import signal
import threading

import websockets

from .config import BrainConfig
from .llm import LlamaServer
from .session import BrainSession
from .stt import STT
from .tts import make_tts
from .voices import VoiceError, VoiceStore, decode_audio

log = logging.getLogger("reachy_brain")


class Brain:
    def __init__(self, cfg: BrainConfig, llm=None, stt=None, tts=None):
        self.cfg = cfg
        self.llm = llm or LlamaServer(cfg)
        self.stt, self.tts = stt, tts
        self.voices: VoiceStore | None = None
        self.active: BrainSession | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        log.info("loading models...")
        if isinstance(self.llm, LlamaServer):
            self.llm.start()
        self.stt = self.stt or STT(self.cfg.stt_model, self.cfg.stt_threads)
        self.tts = self.tts or make_tts(self.cfg.tts_engine, self.cfg.tts_voice, self.cfg.tts_threads)
        self.voices = self.voices or VoiceStore(self.cfg.voices_dir)
        try:                                   # speak in the voice that was chosen last time
            spec = self.voices.spec()
            if spec is not None:
                self.tts.set_voice(spec)
                log.info("voice: %s", self.voices.active())
        except Exception:
            log.exception("could not load the chosen voice; using the default one")
        log.info("brain ready on ws://%s:%d", self.cfg.host, self.cfg.port)

    # ---------- voices (the control page's Voices tab) ----------
    def voice_op(self, m: dict) -> dict:
        """Run one voice request from the robot app and return the reply message. Never raises."""
        t, rid = m.get("type"), m.get("rid")
        ok = lambda **kw: {"type": "voice_result", "rid": rid, "ok": True, **kw}
        bad = lambda msg: {"type": "voice_result", "rid": rid, "ok": False, "error": msg}
        try:
            v = self.voices
            if v is None or self.tts is None:
                return bad("The brain is still starting. Try again in a moment.")
            if t == "voice_list":
                return ok(voices=v.list(), active=v.active(), can_clone=bool(getattr(self.tts, "can_clone", False)))
            if t == "voice_create":
                if not getattr(self.tts, "can_clone", False):
                    return bad("Voice cloning is not set up on the brain. The Kyutai voice model needs its terms accepted "
                               "on Hugging Face and a read login on the brain computer; see the README.")
                meta = v.create(m.get("name"), m.get("consent"), decode_audio(m.get("audio_b64")), m.get("filename"),
                                self.tts.clone)
                return ok(voice=meta, voices=v.list(), active=v.active())
            if t == "voice_use":
                vid = str(m.get("id") or "")
                v.set_active(vid)
                self.tts.set_voice(v.spec(vid))
                return ok(active=v.active(), voices=v.list())
            if t == "voice_delete":
                vid = str(m.get("id") or "")
                was_active = v.active() == vid
                if not v.delete(vid):
                    return bad("That voice is not on the brain any more.")
                if was_active:
                    self.tts.set_voice(None)
                return ok(voices=v.list(), active=v.active())
            return bad("Unknown voice request.")
        except VoiceError as e:
            return bad(str(e))
        except Exception as e:
            log.exception("voice request failed")
            return bad(f"The voice request failed on the brain ({type(e).__name__}).")

    async def _voice_task(self, m: dict, send, sess) -> None:
        res = await asyncio.to_thread(self.voice_op, m)
        send(json.dumps(res))
        if res.get("ok") and m.get("type") == "voice_use" and m.get("preview", True):
            sess.on_say({"text": "Hello. This is how I sound now."})

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
                elif t == "say":
                    sess.on_say(m)
                elif isinstance(t, str) and t.startswith("voice_"):
                    asyncio.create_task(self._voice_task(m, _send, sess))
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

    def _on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_term)       # systemd/kill: shut down cleanly and stop llama-server
    brain = Brain(cfg)
    try:
        brain.load()
    except BaseException:
        brain.llm.stop()
        raise
    try:
        asyncio.run(brain.serve())
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        brain.llm.stop()


if __name__ == "__main__":
    main()
