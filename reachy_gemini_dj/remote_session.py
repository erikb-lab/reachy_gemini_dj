"""Voice session that talks to a local "brain" server (STT + LLM + TTS) over a WebSocket.

Implements the same surface as GeminiSession (start/stop/init_client/push_mic/state/last_error and the
on_* callbacks) so main.py and AudioPipeline do not care which one is active. Robot-side tools are executed
here with the shared execute_tool function; the brain runs web search, weather, camera reading and Gemini help.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from .tool_schemas import robot_tool_specs

log = logging.getLogger(__name__)

TURN_TIMEOUT_S = 90.0


class RemoteSession:
    name = "local"

    def __init__(self, config, execute_tool: Callable[[str, dict], dict], on_event: Callable[[str, str], None],
                 on_voice_chunk: Callable[[bytes], None] | None = None, on_barge_in: Callable[[], None] | None = None,
                 on_heard: Callable[[str], None] | None = None, on_said: Callable[[str], None] | None = None,
                 is_speaking: Callable[[], bool] | None = None, get_frame: Callable[[], Any] | None = None,
                 tool_specs: Callable[[], list[dict]] | None = None):
        self.config = config
        self.execute_tool = execute_tool
        self.on_event = on_event
        self.on_voice_chunk = on_voice_chunk or (lambda b: None)
        self.on_barge_in = on_barge_in or (lambda: None)
        self.on_heard = on_heard or (lambda t: None)
        self.on_said = on_said or (lambda t: None)
        self.is_speaking = is_speaking or (lambda: False)
        self.get_frame = get_frame
        self.tool_specs = tool_specs or robot_tool_specs
        self.state = "starting"
        self.last_error = ""
        self.connected = False
        self.consecutive_failures = 0       # failed connects or failed/timed-out turns since the last good turn
        self.last_connect_ok = 0.0
        self.info: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._out: asyncio.Queue | None = None
        self._restart: asyncio.Event | None = None
        self._turn_started: float | None = None

    # ---------- public surface (same as GeminiSession) ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True, name="brain-client")
        self._thread.start()

    def stop(self, join: bool = True) -> None:
        self._stop.set()
        if self._loop and self._restart:
            self._loop.call_soon_threadsafe(self._restart.set)
        if join and self._thread and self._thread is not threading.current_thread():
            self._thread.join(5)
        self.connected = False
        self.state = "stopped"

    def init_client(self) -> bool:
        """Reconnect (new conversation / settings changed)."""
        if self._loop and self._restart:
            self._loop.call_soon_threadsafe(self._restart.set)
        return True

    def push_mic(self, pcm_16k: bytes) -> None:
        if not (self.connected and self._loop and self._out):
            return
        if self.is_speaking():
            # AudioPipeline already attenuates echo while the robot talks; send silence unless the user is clearly
            # talking over it, so the brain's VAD does not cut the robot off with its own voice.
            a = np.frombuffer(pcm_16k[: len(pcm_16k) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
            if a.size and float(np.sqrt(np.mean(a * a))) < 0.02:
                pcm_16k = bytes(len(pcm_16k))
        self._loop.call_soon_threadsafe(self._put, pcm_16k)

    def say(self, text: str) -> None:
        """Speak an announcement in the local voice (no LLM round)."""
        self._send_json_threadsafe({"type": "say", "text": text})

    def new_conversation(self) -> None:
        self._send_json_threadsafe({"type": "new_conversation"})

    @property
    def turn_overdue(self) -> bool:
        return self._turn_started is not None and time.monotonic() - self._turn_started > TURN_TIMEOUT_S

    # ---------- internals ----------
    def _put(self, item) -> None:
        q = self._out
        if q is None:
            return
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)

    def _send_json_threadsafe(self, obj: dict) -> None:
        if self._loop and self._out:
            self._loop.call_soon_threadsafe(self._put, json.dumps(obj))

    def _hello(self) -> dict:
        c = self.config
        msg = {"type": "hello", "client": "reachy_gemini_dj", "tools": self.tool_specs(),
               "extra_instructions": c.extra_instructions or "",
               "allow_gemini_assist": bool(getattr(c, "allow_gemini_assist", True))}
        if msg["allow_gemini_assist"] and c.api_key:
            msg["gemini_key"] = c.api_key   # only used by the brain for ask_gemini; stays on your network
        return msg

    async def _main(self) -> None:
        import websockets
        self._loop = asyncio.get_running_loop()
        self._out = asyncio.Queue(maxsize=200)
        self._restart = asyncio.Event()
        backoff = 1.0
        while not self._stop.is_set():
            url = (getattr(self.config, "brain_url", "") or "ws://127.0.0.1:8765").strip()
            self.state = "connecting"
            try:
                async with websockets.connect(url, max_size=8 * 1024 * 1024, open_timeout=5,
                                              ping_interval=20, ping_timeout=20) as ws:
                    self._restart.clear()
                    self.connected = True
                    self.last_connect_ok = time.monotonic()
                    self.last_error = ""
                    backoff = 1.0
                    while not self._out.empty():
                        self._out.get_nowait()
                    await ws.send(json.dumps(self._hello()))
                    self.on_event("brain", f"Connected to local brain at {url}")
                    tasks = [asyncio.create_task(self._send_loop(ws)), asyncio.create_task(self._recv_loop(ws)),
                             asyncio.create_task(self._restart.wait())]
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for d in done:
                            d.result()
                    finally:
                        for t in tasks:
                            t.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                if self.last_connect_ok:          # a drop after we had a brain; startup is handled by the grace rule
                    self.consecutive_failures += 1
                self.last_error = f"brain: {type(e).__name__}: {e}"
                self.state = "error"
                log.warning("Brain connection problem: %s", e)
            finally:
                self.connected = False
                self._turn_started = None
                self.on_barge_in()
            if not self._stop.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _send_loop(self, ws) -> None:
        while True:
            item = await self._out.get()
            if isinstance(item, (bytes, bytearray)):
                buf = bytes(item)                       # batch ~40 ms of audio per frame
                while len(buf) < 1280 and not self._out.empty():
                    nxt = self._out.get_nowait()
                    if not isinstance(nxt, (bytes, bytearray)):
                        await ws.send(buf)
                        buf = b""
                        await ws.send(nxt)
                        break
                    buf += nxt
                if buf:
                    await ws.send(buf)
            else:
                await ws.send(item)

    async def _recv_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                self.on_voice_chunk(msg)
                continue
            m = json.loads(msg)
            t = m.get("type")
            if t == "state":
                s = m.get("state", "")
                self.state = s
                if s == "thinking" and self._turn_started is None:
                    self._turn_started = time.monotonic()
            elif t == "heard":
                self.on_heard(m.get("text", ""))
            elif t == "said":
                self.on_said(m.get("text", ""))
            elif t == "clear":
                self.on_barge_in()
            elif t == "user_start":
                if self.is_speaking():
                    self.on_barge_in()
            elif t == "tool_call":
                asyncio.create_task(self._run_tool(ws, m))
            elif t == "request_frame":
                asyncio.create_task(self._send_frame(ws, m.get("id", "")))
            elif t == "event":
                self.on_event(m.get("kind", "brain"), m.get("text", ""))
            elif t == "ready":
                self.info = m
                self.on_event("brain", "Brain ready: " + ", ".join(m.get("tools", [])))
            elif t == "error":
                self.consecutive_failures += 1
                self.last_error = m.get("text", "brain error")
                self.on_event("error", self.last_error)
            elif t == "turn_done":
                if self._turn_started is not None and not self.last_error:
                    self.consecutive_failures = 0
                self._turn_started = None
                self.last_error = ""

    async def _run_tool(self, ws, m: dict) -> None:
        try:
            res = await asyncio.to_thread(self.execute_tool, m.get("name", ""), m.get("args") or {})
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        try:
            await ws.send(json.dumps({"type": "tool_result", "id": m.get("id"), "result": res}, default=str))
        except Exception:
            pass

    async def _send_frame(self, ws, cid: str) -> None:
        jpeg = ""
        try:
            if self.get_frame:
                frame = await asyncio.to_thread(self.get_frame)
                jpeg = encode_jpeg_b64(frame) if frame is not None else ""
        except Exception as e:
            log.info("camera frame unavailable: %s", e)
        await ws.send(json.dumps({"type": "frame", "id": cid, "jpeg_b64": jpeg}))


def encode_jpeg_b64(frame, max_side: int = 640) -> str:
    """BGR/RGB numpy image -> base64 JPEG, downscaled so the vision model reads it quickly."""
    try:
        import cv2
        h, w = frame.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode() if ok else ""
    except ImportError:
        import io
        from PIL import Image
        img = Image.fromarray(np.asarray(frame)[..., ::-1])
        img.thumbnail((max_side, max_side))
        b = io.BytesIO()
        img.save(b, "JPEG", quality=80)
        return base64.b64encode(b.getvalue()).decode()
