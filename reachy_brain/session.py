"""One robot connection: mic audio in -> VAD -> STT -> LLM with tools -> TTS audio out.

Transport-agnostic: the server gives the session two callables, send_json(dict) and send_audio(bytes), that are safe
to call from any thread. Robot-side tools (music, motion, calendar...) are executed by the robot: the session sends
{"type": "tool_call"} and blocks this turn's worker thread until the matching {"type": "tool_result"} arrives.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import queue
import re
import threading
import time
import uuid
from typing import Any, Callable

import numpy as np

from .assist import SERVER_TOOLS, run_server_tool
from .llm import Cancelled
from .segmenter import Segmenter

log = logging.getLogger(__name__)

INFO_SERVER_TOOLS = set(SERVER_TOOLS)

# Spoken instantly after a successful action tool when the model said nothing itself. Saves a whole extra LLM
# round (several seconds on a CPU) that would only produce "ok, done".
CANNED_ACK = {
    "play_music": "Okay, queuing that up.", "pause_music": "Paused.", "resume_music": "Back to the music.",
    "skip_music": "Skipping.", "stop_music": "Music stopped.", "set_volume": "Volume adjusted.",
    "robot_emotion": "There you go.", "robot_dance": "Watch this!", "robot_look": "Looking.",
    "robot_pose": "Moving.", "save_pose": "Pose saved.", "calendar_add": "Added to your calendar.",
    "tasks_add": "Added to your tasks.", "tasks_complete": "Marked as done.",
}


# "play/queue something": the small model sometimes answers "sure, playing it" without calling the tool at all.
# For these utterances the first round must be a tool call (spoken confirmation comes from CANNED_ACK).
_MUSIC_INTENT = re.compile(r"\b(play|queue|put on|spin|throw on)\b(?!.*\b(game|role|trick|joke|part)\b)", re.I)


# What Whisper "hears" in silence or room noise; never a real request.
GHOST_TRANSCRIPTS = {"you", "thank you", "thanks", "thanks for watching", "bye", "bye bye", "the end"}


# Music transport commands need no language model: match them directly (instant, and never mapped to the wrong tool).
_QUICK = [
    ("pause_music", "Paused.", re.compile(r"^(please |can you |could you )*(pause|hold)( the| this| that)?( music| song| track| it)?( please)?$")),
    ("resume_music", "Back to the music.", re.compile(
        r"^(please |can you |could you )*(resume|unpause|continue|keep going|play)( the| this| that)?"
        r"( music| song| track| it)?( again| back)?( please)?$")),
    ("skip_music", "Skipping.", re.compile(r"^(please |can you |could you )*(skip|next)( the| this| that)?( music| song| track| one)?( please)?$")),
    ("stop_music", "Music stopped.", re.compile(r"^(please |can you |could you )*stop( the| this| that)?( music| song| track| playing)( please)?$")),
]


def quick_command(heard: str) -> tuple[str, str] | None:
    """('tool_name', spoken ack) when the whole utterance is a bare music transport command, else None."""
    t = re.sub(r"[^a-z' ]", " ", heard.lower())
    t = " ".join(t.split())
    for name, ack, rx in _QUICK:
        if rx.match(t):
            return name, ack
    return None


def _failed(res: Any) -> bool:
    return not isinstance(res, dict) or "error" in res


class BrainSession:
    def __init__(self, cfg, llm, stt, tts, send_json: Callable[[dict], None], send_audio: Callable[[bytes], None]):
        self.cfg, self.llm, self.stt, self.tts = cfg, llm, stt, tts
        self.send_json, self.send_audio = send_json, send_audio
        self.seg = Segmenter(cfg.vad_aggressiveness, cfg.end_silence_ms, cfg.max_utterance_s)
        self.robot_tools: list[dict] = []          # OpenAI-format schemas sent by the robot
        self.robot_kinds: dict[str, str] = {}      # name -> "info" | "action"
        self.extra_instructions = ""
        self.gemini_key = cfg.gemini_api_key
        self.allow_assist = True
        self.vision = bool(cfg.mmproj_path)
        self.history: list[dict] = []
        self._utts: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._busy = False
        self._speaking = False
        self._pending: dict[str, tuple[threading.Event, list]] = {}
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="brain-turns")
        self._worker.start()

    # ---------- inbound from the robot ----------
    def on_hello(self, msg: dict) -> None:
        tools = msg.get("tools") or []
        self.robot_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}}}} for t in tools]
        self.robot_kinds = {t["name"]: t.get("kind", "action") for t in tools}
        self.extra_instructions = msg.get("extra_instructions") or ""
        self.allow_assist = bool(msg.get("allow_gemini_assist", True))
        if msg.get("gemini_key") and not self.cfg.gemini_api_key:
            self.gemini_key = msg["gemini_key"]
        self.send_json({"type": "ready", "vision": self.vision, "assist": self._assist_on(),
                        "tools": [t["function"]["name"] for t in self._tools()]})
        self._utts.put("warm")   # fill the LLM prompt cache with this system prompt + tool list

    def on_audio(self, pcm: bytes) -> None:
        for kind, payload in self.seg.feed(pcm):
            if kind == "start":
                self.send_json({"type": "user_start"})
                if self._busy and self._speaking:
                    self.barge_in()          # only interrupt once Reachy is actually talking; a blip while it is
                                             # still transcribing/thinking must not throw the question away
            elif kind == "utterance":
                self._utts.put(payload)

    def on_tool_result(self, msg: dict) -> None:
        slot = self._pending.get(msg.get("id", ""))
        if slot:
            slot[1].append(msg.get("result"))
            slot[0].set()

    def on_frame(self, msg: dict) -> None:
        self.on_tool_result({"id": msg.get("id", ""), "result": msg.get("jpeg_b64", "")})

    def on_say(self, msg: dict) -> None:
        """Robot-initiated announcement (e.g. 'we are queuing'): spoken as-is, after any turn in progress."""
        text = str(msg.get("text") or "").strip()[:400]
        if text:
            self._utts.put(("say", text))

    def barge_in(self) -> None:
        if self._busy:
            log.info("barge-in: cancelling the current turn")
            self._cancel.set()
            self.send_json({"type": "clear"})

    def new_conversation(self) -> None:
        self.barge_in()
        self.history.clear()

    def close(self) -> None:
        self._closed.set()
        self._cancel.set()
        for ev, _ in list(self._pending.values()):
            ev.set()
        self._utts.put(None)

    # ---------- helpers ----------
    def _state(self, s: str, detail: str = "") -> None:
        self.send_json({"type": "state", "state": s, "detail": detail})

    def _assist_on(self) -> bool:
        return bool(self.allow_assist and self.gemini_key)

    def _tools(self) -> list[dict]:
        tools = list(self.robot_tools)
        for name, schema in SERVER_TOOLS.items():
            if name == "ask_gemini" and not self._assist_on():
                continue
            if name == "look" and not self.vision:
                continue
            tools.append(schema)
        return tools

    def _system(self) -> str:
        today = datetime.date.today().strftime("%A %Y-%m-%d")
        s = f"{self.cfg.persona} Today is {today}."
        if self._assist_on():
            s += " For hard questions (math, code, multi-step reasoning, long explanations) call ask_gemini, then " \
                 "say its answer. If the user says 'ask Gemini', always use it."
        if self.extra_instructions:
            s += "\n\nUser context:\n" + self.extra_instructions
        return s

    def _robot_call(self, name: str, args: dict, timeout: float) -> Any:
        cid = uuid.uuid4().hex[:12]
        ev, box = threading.Event(), []
        self._pending[cid] = (ev, box)
        try:
            if name == "__frame__":
                self.send_json({"type": "request_frame", "id": cid})
            else:
                self.send_json({"type": "tool_call", "id": cid, "name": name, "args": args})
            ev.wait(timeout)
            if self._closed.is_set():
                raise Cancelled()
            return box[0] if box else {"error": f"The robot did not answer the {name} call in time."}
        finally:
            self._pending.pop(cid, None)

    def _look(self, question: str) -> dict:
        jpeg = self._robot_call("__frame__", {}, 8.0)
        if not isinstance(jpeg, str) or not jpeg:
            return {"error": "No camera image is available."}
        try:
            return {"seen": self.llm.vision_answer(jpeg, question)}
        except Exception as e:
            return {"error": f"Could not read the image ({type(e).__name__})."}

    def _speak(self, text: str) -> None:
        """Synthesize one clause and stream it to the robot as 24 kHz int16 PCM."""
        if self._cancel.is_set() or not text.strip():
            return
        self.send_json({"type": "said", "text": text})
        self._speaking = True
        for chunk in self.tts.stream(text):
            if self._cancel.is_set():
                return
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            self.send_audio(pcm)

    # ---------- turn worker ----------
    def _run(self) -> None:
        while not self._closed.is_set():
            audio = self._utts.get()
            if audio is None:
                return
            if isinstance(audio, str):          # "warm"
                self._warm()
                continue
            # drop stale utterances if several queued up while busy: answer the newest only
            while isinstance(audio, np.ndarray) and not self._utts.empty():
                nxt = self._utts.queue[0]
                if not isinstance(nxt, np.ndarray):
                    break
                audio = self._utts.get_nowait()
            self._cancel.clear()
            self._busy = True
            try:
                if isinstance(audio, tuple):      # ("say", text)
                    self._state("speaking")
                    self._speak(audio[1])
                    self.history += [{"role": "assistant", "content": audio[1]}]
                    self.history = self.history[-2 * self.cfg.history_turns:]
                else:
                    self._turn(audio)
            except Cancelled:
                log.info("turn cancelled")
            except Exception as e:
                log.exception("turn failed")
                self.send_json({"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                self._busy = False
                self._speaking = False
                if not self._closed.is_set():
                    self.send_json({"type": "turn_done"})
                    self._state("listening")

    def _warm(self) -> None:
        """Process the system prompt and tool schemas once so the first real question is fast (otherwise the
        first turn re-reads ~2k prompt tokens on a CPU, which took ~170 s in testing).
        llama-server keeps its prompt cache across connections, so a reconnect with the same prompt skips this."""
        key = hashlib.sha1(json.dumps([self._system(), self._tools()], sort_keys=True).encode()).hexdigest()
        if getattr(self.llm, "warm_key", None) == key:
            log.info("LLM already warm for this prompt; skipping warm-up")
            return
        self._state("loading", "warming up the language model")
        t0 = time.time()
        try:
            msgs = [{"role": "system", "content": self._system()}, {"role": "user", "content": "hi"}]
            self.llm.chat_stream(msgs, self._tools(), lambda c: None, threading.Event(), max_tokens=1)
            self.llm.warm_key = key
            log.info("LLM warm-up took %.1fs", time.time() - t0)
        except Exception as e:
            log.warning("LLM warm-up failed: %s", e)
        self._state("listening")

    def _turn(self, audio: np.ndarray) -> None:
        t0 = time.time()
        self._state("thinking", "transcribing")
        heard = self.stt.transcribe(audio)
        if not heard or len(heard.strip(" .,!?")) < 2 or heard.strip(" .,!?").lower() in GHOST_TRANSCRIPTS:
            return
        self.send_json({"type": "heard", "text": heard})
        quick = quick_command(heard)
        if quick and any(t["function"]["name"] == quick[0] for t in self.robot_tools):
            name, ack = quick
            self.send_json({"type": "event", "kind": "tool", "text": f"{name} (voice shortcut)"})
            res = self._robot_call(name, {}, self.cfg.tool_timeout_s)
            say = ack if not _failed(res) else str((res or {}).get("error", "That didn't work."))[:120]
            self._state("speaking")
            self._speak(say)
            self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": say}]
            self.history = self.history[-2 * self.cfg.history_turns:]
            log.info("turn: heard=%r shortcut=%s total=%.1fs", heard, name, time.time() - t0)
            return
        msgs = [{"role": "system", "content": self._system()}, *self.history, {"role": "user", "content": heard}]
        tools = self._tools()
        choice, spoken_all, first_audio = "auto", [], [None]
        if _MUSIC_INTENT.search(heard) and any(t["function"]["name"] == "play_music" for t in tools):
            choice = "required"

        def on_clause(text: str) -> None:
            if first_audio[0] is None:
                first_audio[0] = time.time() - t0
                self._state("speaking")
            spoken_all.append(text)
            self._speak(text)

        for _ in range(3):
            if self._cancel.is_set():
                raise Cancelled()
            content, calls = self.llm.chat_stream(msgs, tools, on_clause, self._cancel, tool_choice=choice)
            if not calls:
                break
            msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
            names, any_failed = [], False
            for c in calls:
                name = c["function"]["name"]
                names.append(name)
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.send_json({"type": "event", "kind": "tool", "text": f"{name} {args}"})
                if name in SERVER_TOOLS:
                    self._state("thinking", name)
                    res = run_server_tool(name, args, gemini_key=self.gemini_key, assist_model=self.cfg.assist_model,
                                          look=self._look)
                else:
                    res = self._robot_call(name, args, self.cfg.tool_timeout_s)
                any_failed = any_failed or _failed(res)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res)[:4000]})
            if choice == "required":
                choice = "auto"                  # forced only for the first round
            if any(n in INFO_SERVER_TOOLS or self.robot_kinds.get(n) == "info" for n in names):
                choice = "none"                  # answer from the result; no second lookup
            elif content.strip():
                break                            # already spoke and only action tools ran: done
            elif not any_failed and all(n in CANNED_ACK for n in names):
                on_clause(CANNED_ACK[names[0]])  # nothing said yet: confirm instantly, skip another LLM round
                break
        reply = " ".join(spoken_all).strip()
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": reply or "(done)"}]
        self.history = self.history[-2 * self.cfg.history_turns:]
        log.info("turn: heard=%r first_audio=%.1fs total=%.1fs", heard,
                 first_audio[0] or -1, time.time() - t0)
