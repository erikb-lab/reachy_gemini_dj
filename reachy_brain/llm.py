"""llama-server (OpenAI-compatible) lifecycle and streaming chat client with clause-level callbacks."""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)

SENTENCE_MIN, CLAUSE_MIN = 12, 40
_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[,;:])\s+")


def split_ready(buf: str) -> tuple[str, str] | None:
    """Return (clause_to_speak, rest) if buf holds a speakable clause, else None.
    Speak at the end of a sentence, or at a comma/semicolon/colon once CLAUSE_MIN chars have accumulated, so a
    long sentence starts being spoken early."""
    for m in _BREAK.finditer(buf):
        need = SENTENCE_MIN if buf[m.start() - 1] in ".!?" else CLAUSE_MIN
        if m.start() >= need:
            return buf[: m.start()].strip(), buf[m.end():]
    return None


def _die_with_parent() -> None:
    """Linux: have the kernel SIGTERM llama-server if the brain process dies, so it never lingers."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


class Cancelled(Exception):
    pass


class LlamaServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.base = cfg.llama_url.rstrip("/") if cfg.llama_url else f"http://127.0.0.1:{cfg.llm_port}"

    def start(self, timeout_s: float = 600.0) -> None:
        c = self.cfg
        if not c.llama_url:
            if not c.model_path:
                raise RuntimeError("no --model (or --llama-url) given for the LLM")
            cmd = [c.llama_server_bin, "-m", c.model_path, "--alias", "brain", "--host", "127.0.0.1",
                   "--port", str(c.llm_port), "-t", str(c.llm_threads), "-c", str(c.llm_ctx * 2), "-np", "2",
                   "--no-kv-unified", "--cache-reuse", "256", "--reasoning", "off", "--no-webui"]
            if c.slot_cache_dir:
                os.makedirs(self.slot_dir(), exist_ok=True)
                cmd += ["--slot-save-path", self.slot_dir()]
            # slot 0 = the conversation, slot 1 = one-off camera questions. Separate KV per slot (--no-kv-unified)
            # is required: with a unified pool the camera request wiped slot 0 (measured: next turn 69 s vs 1.4 s). The
            # conversation's cached prompt; --cache-reuse keeps the cache usable when old turns are trimmed
            if c.mmproj_path:
                cmd += ["--mmproj", c.mmproj_path]
            log.info("starting llama-server: %s", " ".join(cmd))
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
            atexit.register(self.stop)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
            try:
                if httpx.get(self.base + "/health", timeout=2).status_code == 200:
                    log.info("llama-server ready")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.5)
        raise RuntimeError("llama-server did not become ready")

    def slot_dir(self) -> str:
        return os.path.expanduser(self.cfg.slot_cache_dir or "")

    def slot_restore(self, filename: str) -> bool:
        """Load a saved prompt cache into slot 0. False on any failure (missing file, no --slot-save-path...)."""
        return self._slot_action("restore", filename)

    def slot_save(self, filename: str) -> bool:
        """Save slot 0's prompt cache under filename (relative to the server's --slot-save-path)."""
        ok = self._slot_action("save", filename)
        if ok:
            self._prune_slots()
        return ok

    def _slot_action(self, action: str, filename: str) -> bool:
        try:
            r = httpx.post(f"{self.base}/slots/0", params={"action": action}, json={"filename": filename}, timeout=120)
            if r.status_code == 200:
                return True
            log.info("slot %s %s: HTTP %s", action, filename, r.status_code)
        except Exception as e:
            log.info("slot %s %s failed: %s", action, filename, e)
        return False

    def _prune_slots(self, keep: int = 3) -> None:
        """Keep the newest few warm-*.bin files (only when the slot dir is local, i.e. we launched the server)."""
        if not self.proc:
            return
        try:
            d = self.slot_dir()
            files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("warm-") and f.endswith(".bin")),
                           key=os.path.getmtime, reverse=True)
            for f in files[keep:]:
                os.remove(f)
        except OSError as e:
            log.info("could not prune slot cache: %s", e)

    def healthy(self) -> bool:
        try:
            return httpx.get(self.base + "/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def chat_stream(self, messages: list[dict], tools: list[dict], on_clause: Callable[[str], None],
                    cancel: threading.Event, tool_choice: str = "auto",
                    max_tokens: int | None = None) -> tuple[str, list[dict]]:
        """Stream one completion. Returns (spoken_text, tool_calls). on_clause gets each speakable clause."""
        c = self.cfg
        body: dict[str, Any] = {"messages": messages, "temperature": c.temperature, "top_p": c.top_p,
                                "max_tokens": max_tokens or c.max_tokens, "stream": True, "id_slot": 0,
                                "cache_prompt": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        content, buf, calls = "", "", {}
        with httpx.stream("POST", self.base + "/v1/chat/completions", json=body, timeout=httpx.Timeout(900, connect=10)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if cancel.is_set():
                    raise Cancelled()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                ch = d["choices"][0].get("delta", {}) if d.get("choices") else {}
                piece = ch.get("content") or ""
                if piece:
                    content += piece
                    buf += piece
                    while (got := split_ready(buf)):
                        on_clause(got[0])
                        buf = got[1]
                for tc in ch.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    cur = calls.setdefault(i, {"id": tc.get("id") or f"call{i}", "function": {"name": "", "arguments": ""}})
                    f = tc.get("function") or {}
                    cur["function"]["name"] += f.get("name") or ""
                    cur["function"]["arguments"] += f.get("arguments") or ""
        if buf.strip():
            on_clause(buf.strip())
        return content, [dict(type="function", **v) for _, v in sorted(calls.items())]

    def vision_answer(self, jpeg_b64: str, question: str) -> str:
        """One non-streaming call with an image (needs the model started with an mmproj)."""
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64}},
            {"type": "text", "text": question or "Describe what you see in one or two short sentences."}]}],
            "max_tokens": 120, "temperature": 0.3, "id_slot": 1}
        r = httpx.post(self.base + "/v1/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
