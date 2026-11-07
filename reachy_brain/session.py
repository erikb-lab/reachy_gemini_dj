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
from .speakable import speakable

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


# Requests the small model tends to "answer" by making things up instead of calling the tool (seen in the laptop
# end-to-end test: it described an imaginary room, invented a World Series winner and team facts, and said it was
# showing an emotion without moving). For these the first round must be a tool call, and a hint names the tool.
# (The tool list itself is never changed between rounds: that would throw away llama-server's prompt cache.)
_INTENTS: list[tuple[re.Pattern, str]] = [
    # timers and memory first, so "remember ..." / "timer ..." never fall through to web_search
    (re.compile(r"\b(what (do|did) you remember|what have i (asked|told) you to remember|list (my |your )?memor)", re.I),
     "list_memories"),
    (re.compile(r"\bforget (that|about)\b", re.I), "forget"),
    (re.compile(r"\b(remember (that|this|my|our|the|to)\b|don'?t forget|do not forget|please remember)", re.I),
     "remember"),
    (re.compile(r"\bcancel\b.{0,20}\b(timer|alarm)s?\b|\b(delete|clear)\b.{0,20}\btimers?\b", re.I), "cancel_timer"),
    (re.compile(r"\b(stop|turn off|silence|dismiss|shut off)\b.{0,15}\balarm\b", re.I), "stop_alarm"),
    (re.compile(r"\bhow (long|much time)\b.{0,15}\b(left|remaining|on)\b|\bmy timers\b|\b(any|what|which) timers\b", re.I),
     "list_timers"),
    (re.compile(r"\balarm\b|\bwake me\b", re.I), "set_alarm"),
    (re.compile(r"\btimers?\b|\bremind me in\b", re.I), "set_timer"),
    (re.compile(r"\b(what (do|can) you see|what are you (looking at|seeing)|look at (this|that|me|it)|in front of you|"
                r"what('s| is) (this|that) (thing|i'm holding)|read (this|that|it)|can you see)\b", re.I), "look"),
    (re.compile(r"\b(weather|temperature|forecast|raining|snowing)\b", re.I), "get_weather"),
    (re.compile(r"\b(next match|queu(e|ing) (status|now)|when do we play)\b", re.I), "frc_next_match"),
    (re.compile(r"\b(team|frc)\s*\d{1,5}\b|\babout (our|my) team\b", re.I), "frc_team_info"),
    (re.compile(r"\b(ranking|rankings|who is (in )?first)\b", re.I), "frc_rankings"),
    (re.compile(r"\bdance\b", re.I), "robot_dance"),
    (re.compile(r"\b(show|act|look|be|get)\b.{0,25}\b(happy|sad|angry|surprised|excited|scared|tired|confused|"
                r"curious|bored|proud|shy|love)\b", re.I), "robot_emotion"),
    (re.compile(r"\b(who won|latest|news|search|look up|this year|this season|current(ly)?|today'?s)\b", re.I),
     "web_search"),
]


# Spoken instantly (before the LLM round) when a slow tool is forced.
FILLERS = {
    "get_weather": "Let me check the weather.", "web_search": "Let me look that up.", "look": "Let me take a look.",
    "frc_next_match": "Checking the schedule.", "frc_team_info": "Looking up that team.",
    "frc_rankings": "Checking the rankings.",
}


def forced_tool(heard: str, available: set[str]) -> str | None:
    """Name of the tool this request must start with, or None to let the model decide."""
    for rx, name in _INTENTS:
        if rx.search(heard) and name in available:
            return name
    if _MUSIC_INTENT.search(heard) and "play_music" in available:
        return "play_music"
    return None


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


_PLEASE = r"(?:please |can you |could you |hey reachy )*"
_NUM = (r"\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
        r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|forty five|fifty|sixty|ninety")
_UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)"
_DUR = (rf"(?:half an hour|(?:{_NUM})(?: and a half)? {_UNIT}(?: and (?:{_NUM}) {_UNIT})?|"
        rf"an hour and a half)")
_QUICK_TIMER = [re.compile(rf"^{_PLEASE}(?:set|start|make|create)(?: me)?(?: a| an| the)? timer (?:for|of) (?P<dur>{_DUR})$"),
                re.compile(rf"^{_PLEASE}(?:set|start|make|create)(?: me)?(?: a| an| the)? (?P<dur>{_DUR}) timer$"),
                re.compile(rf"^{_PLEASE}(?P<dur>{_DUR}) timer$")]
_QUICK_CANCEL = re.compile(rf"^{_PLEASE}(?:cancel|delete|clear)(?: the| my| all| all of my| all my)? (?P<t>timers?)$")
_QUICK_STOP_ALARM = re.compile(rf"^{_PLEASE}(?:stop|turn off|silence|dismiss|shut off)(?: the| my)? alarm$")


def quick_command(heard: str) -> tuple[str, str, dict] | None:
    """('tool_name', spoken ack, args) when the whole utterance is a bare music transport / timer command, else None."""
    t = re.sub(r"[^a-z0-9.' ]", " ", heard.lower())
    t = " ".join(t.split()).rstrip(". ")
    for name, ack, rx in _QUICK:
        if rx.match(t):
            return name, ack, {}
    for rx in _QUICK_TIMER:
        if m := rx.match(t):
            return "set_timer", "Done.", {"duration": m.group("dur")}
    if m := _QUICK_CANCEL.match(t):
        return "cancel_timer", "Done.", {"which": "all" if m.group("t") == "timers" else ""}
    if _QUICK_STOP_ALARM.match(t):
        return "stop_alarm", "Done.", {}
    return None


def _compact(res: Any) -> str:
    """What history keeps for a tool result the LLM never read (the spoken answer is in the next message)."""
    if _failed(res):
        return json.dumps({"error": str((res or {}).get("error", "failed") if isinstance(res, dict) else "failed")[:120]})
    return '{"ok": true}'


def _compact_unsent(msgs: list[dict], results: list[tuple[int, Any]]) -> None:
    """Results answered without another LLM round were never sent to llama-server, so they are not in its prompt
    cache: storing them whole would make the NEXT turn read them (~20 tokens/s on a CPU). Keep a stub instead."""
    for i, res in results:
        msgs[i] = {**msgs[i], "content": _compact(res)}


def say_of(res: Any) -> str:
    """The ready-to-speak answer in a tool result ({"say": "..."} with no error), else ''."""
    if isinstance(res, dict) and "error" not in res and isinstance(res.get("say"), str):
        return res["say"].strip()
    return ""


_WORDS = re.compile(r"[a-z0-9']+")


def looks_hallucinated(text: str) -> bool:
    """Whisper on noise or music often loops one or two words ("yes, cool. yes, cool. yes, cool.")."""
    words = _WORDS.findall(text.lower())
    return len(words) >= 6 and len(set(words)) / len(words) < 0.45


def echo_overlap(heard: str, said: str) -> float:
    """Share of the heard words that Reachy itself just said (its own voice picked up by the mic)."""
    h = [w for w in _WORDS.findall(heard.lower()) if len(w) > 2]
    if len(h) < 3:
        return 0.0
    s = set(_WORDS.findall(said.lower()))
    return sum(w in s for w in h) / len(h)


BARGE_IN_VOICED_FRAMES = 12   # ~360 ms of speech before Reachy stops talking: a blip of its own echo is not enough


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
        self.memory: list[str] = []                 # facts the robot was asked to remember (set at hello only)
        self.gemini_key = cfg.gemini_api_key
        self.allow_assist = True
        self.robot_assist = False                   # the robot calls Gemini itself (its key never comes here)
        self.vision = bool(cfg.mmproj_path)
        self.turns: list[list[dict]] = []          # past turns, each the full message list incl. tool calls
        self._utts: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._busy = False
        self._speaking = False
        self._play_end = 0.0                        # when the robot finishes playing what was sent (monotonic)
        self._said_recent: list[tuple[float, str]] = []   # (time, text) Reachy spoke, for echo rejection
        self._barge_pending = False
        self._utt_start = 0.0
        self._utt_starts: dict[int, float] = {}
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
        self.memory = [str(f).strip() for f in (msg.get("memory") or []) if str(f).strip()]
        self.allow_assist = bool(msg.get("allow_gemini_assist", True))
        self.robot_assist = bool(msg.get("assist_on_robot"))
        if msg.get("gemini_key") and not self.cfg.gemini_api_key:
            self.gemini_key = msg["gemini_key"]
        self.send_json({"type": "ready", "vision": self.vision, "assist": self._assist_on(),
                        "tools": [t["function"]["name"] for t in self._tools()]})
        self._utts.put("warm")   # fill the LLM prompt cache with this system prompt + tool list

    def on_audio(self, pcm: bytes) -> None:
        for kind, payload in self.seg.feed(pcm):
            if kind == "start":
                self.send_json({"type": "user_start"})
                self._utt_start = time.monotonic()
                # only interrupt once Reachy is actually talking (a blip while it is still transcribing/thinking
                # must not throw the question away), and only after real speech, not a short echo of its own voice
                self._barge_pending = self._busy and self._speaking
            elif kind == "utterance":
                self._barge_pending = False
                self._utt_starts[id(payload)] = self._utt_start
                self._utts.put(payload)
            elif kind == "cancel":
                self._barge_pending = False
        if self._barge_pending and self.seg.speaking and self.seg.voiced_frames >= BARGE_IN_VOICED_FRAMES:
            self._barge_pending = False
            if self._busy and self._speaking:
                self.barge_in()

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
        self.turns.clear()

    def close(self) -> None:
        self._closed.set()
        self._cancel.set()
        for ev, _ in list(self._pending.values()):
            ev.set()
        self._utts.put(None)

    # ---------- helpers ----------
    def _state(self, s: str, detail: str = "") -> None:
        self.send_json({"type": "state", "state": s, "detail": detail})

    @property
    def history(self) -> list[dict]:
        return [m for t in self.turns for m in t]

    def _remember(self, msgs: list[dict]) -> None:
        """Keep a finished turn exactly as it was sent (tool calls and hints included): any change to past
        messages would invalidate llama-server's prompt cache and cost tens of seconds per turn on a CPU.
        Old turns are dropped in one go (down to 60% of the budget) so the cache breaks rarely."""
        self.turns.append([dict(m) for m in msgs])
        size = lambda: sum(len(json.dumps(m)) for t in self.turns for m in t)
        if size() > self.cfg.history_chars:
            while len(self.turns) > 1 and size() > self.cfg.history_chars * 0.6:
                self.turns.pop(0)

    def _assist_on(self) -> bool:
        return bool(self.allow_assist and (self.gemini_key or self.robot_assist))

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
            s += " For hard questions or when asked to ask Gemini, use ask_gemini and say its answer."
        if self.extra_instructions:
            s += "\n\nUser context:\n" + self.extra_instructions
        return s

    def _pinned(self) -> list[dict]:
        """Remembered facts as a fixed pair right after the system message (never in the system text itself, which
        would invalidate the cached tools prefix; never trimmed)."""
        if not self.memory:
            return []
        return [{"role": "user", "content": "Things you were asked to remember: " + "; ".join(self.memory)},
                {"role": "assistant", "content": "Noted."}]

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
        now = time.monotonic()
        self._said_recent = [(t, x) for t, x in self._said_recent if now - t < 30] + [(now, text)]
        for chunk in self.tts.stream(speakable(text)):
            if self._cancel.is_set():
                return
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            self.send_audio(pcm)
            self._play_end = max(time.monotonic(), self._play_end) + len(pcm) / 2 / 24000

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
                    self._remember([{"role": "assistant", "content": audio[1]}])
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
        key = hashlib.sha1(json.dumps([self._system(), self._tools(), self._pinned()], sort_keys=True).encode()).hexdigest()
        if getattr(self.llm, "warm_key", None) == key:
            log.info("LLM already warm for this prompt; skipping warm-up")
            return
        self._state("loading", "warming up the language model")
        t0 = time.time()
        fname = f"warm-{key[:16]}.bin"
        try:
            if hasattr(self.llm, "slot_restore") and self.llm.slot_restore(fname):
                self.llm.warm_key = key
                log.info("LLM prompt cache restored from disk in %.1fs", time.time() - t0)
                self._state("listening")
                return
        except Exception as e:
            log.info("prompt cache restore failed: %s", e)
        try:
            msgs = [{"role": "system", "content": self._system()}, *self._pinned(), {"role": "user", "content": "hi"}]
            self.llm.chat_stream(msgs, self._tools(), lambda c: None, threading.Event(), max_tokens=1)
            self.llm.warm_key = key
            log.info("LLM warm-up took %.1fs", time.time() - t0)
        except Exception as e:
            log.warning("LLM warm-up failed: %s", e)
        else:
            try:
                if hasattr(self.llm, "slot_save"):
                    self.llm.slot_save(fname)
            except Exception as e:
                log.info("prompt cache save failed: %s", e)
        self._state("listening")

    def _reject_reason(self, heard: str, started: float) -> str:
        """Why a transcript is not a real request ("" = it is). Ghost phrases and word loops come from Whisper
        hearing noise or music; echo is Reachy's own voice picked up by the mic while it was talking."""
        core = (heard or "").strip(" .,!?")
        if len(core) < 2:
            return "empty"
        if core.lower() in GHOST_TRANSCRIPTS:
            return "ghost phrase"
        if looks_hallucinated(heard):
            return "repeated words"
        if started and started < self._play_end + 1.0:
            said = " ".join(x for t, x in self._said_recent if t >= started - 30)
            if echo_overlap(heard, said) >= 0.6:
                return "echo of Reachy's own voice"
        return ""

    def _turn(self, audio: np.ndarray) -> None:
        t0 = time.time()
        self._state("thinking", "transcribing")
        started = self._utt_starts.pop(id(audio), 0.0)
        heard = self.stt.transcribe(audio)
        reason = self._reject_reason(heard, started)
        log.info("stt: %r%s", heard, f" -> ignored ({reason})" if reason else "")
        if reason:
            return
        self.send_json({"type": "heard", "text": heard})
        quick = quick_command(heard)
        if quick and any(t["function"]["name"] == quick[0] for t in self.robot_tools):
            name, ack, qargs = quick
            self.send_json({"type": "event", "kind": "tool", "text": f"{name} (voice shortcut)"})
            res = self._robot_call(name, qargs, self.cfg.tool_timeout_s)
            say = (say_of(res) or ack) if not _failed(res) else str((res or {}).get("error", "That didn't work."))[:120]
            self._state("speaking")
            self._speak(say)
            cid = "q" + uuid.uuid4().hex[:8]
            self._remember([{"role": "user", "content": heard},
                            {"role": "assistant", "content": "", "tool_calls": [
                                {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(qargs)}}]},
                            {"role": "tool", "tool_call_id": cid, "content": _compact(res)},
                            {"role": "assistant", "content": say}])
            log.info("turn: heard=%r shortcut=%s total=%.1fs", heard, name, time.time() - t0)
            return
        tools = self._tools()
        choice, spoken_all, first_audio = "auto", [], [None]
        force = forced_tool(heard, {t["function"]["name"] for t in tools})
        user_msg = {"role": "user", "content": heard}
        if force:
            choice = "required"
            user_msg = {"role": "user", "content": f"{heard}\n(Call {force} first. Do not answer from memory.)"}
        msgs = [{"role": "system", "content": self._system()}, *self._pinned(), *self.history, user_msg]
        base = len(msgs) - 1

        def on_clause(text: str) -> None:
            if first_audio[0] is None:
                first_audio[0] = time.time() - t0
                self._state("speaking")
            spoken_all.append(text)
            self._speak(text)

        if force in FILLERS:
            # instant filler while the slow tool runs; not in history (history must equal what the LLM was sent).
            # _speaking goes back to False so a cough while the model thinks does not cancel the turn.
            self._state("speaking")
            self._speak(FILLERS[force])
            self._speaking = False
        final = ""
        for _ in range(3):
            if self._cancel.is_set():
                raise Cancelled()
            content, calls = self.llm.chat_stream(msgs, tools, on_clause, self._cancel, tool_choice=choice)
            if not calls:
                final = content
                break
            msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
            names, any_failed, says, results = [], False, [], []
            for c in calls:
                name = c["function"]["name"]
                names.append(name)
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.send_json({"type": "event", "kind": "tool", "text": f"{name} {args}"})
                if name == "ask_gemini" and not self.gemini_key and self.robot_assist:
                    self._state("thinking", name)
                    res = self._robot_call(name, args, 60.0)      # robot asks Gemini with its own key
                elif name in SERVER_TOOLS:
                    self._state("thinking", name)
                    res = run_server_tool(name, args, gemini_key=self.gemini_key, assist_model=self.cfg.assist_model,
                                          look=self._look)
                else:
                    res = self._robot_call(name, args, self.cfg.tool_timeout_s)
                any_failed = any_failed or _failed(res)
                says.append(say_of(res))             # before truncation: the spoken answer must stay whole
                results.append((len(msgs), res))
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res)[:self.cfg.tool_result_chars]})
            if all(says):
                text = " ".join(says)            # every tool gave a ready answer: speak it, no more LLM rounds
                _compact_unsent(msgs, results)
                on_clause(text)
                final = text
                break
            if choice == "required":
                choice = "auto"                  # forced only for the first round
            if any(n in INFO_SERVER_TOOLS or self.robot_kinds.get(n) == "info" for n in names):
                choice = "none"                  # answer from the result; no second lookup
            elif content.strip():
                break                            # already spoke and only action tools ran: done
            elif not any_failed and all(n in CANNED_ACK for n in names):
                on_clause(CANNED_ACK[names[0]])  # nothing said yet: confirm instantly, skip another LLM round
                _compact_unsent(msgs, results)
                final = CANNED_ACK[names[0]]
                break
        reply = " ".join(spoken_all).strip()
        turn = msgs[base:]
        turn.append({"role": "assistant", "content": final or reply or "(done)"})
        self._remember(turn)
        log.info("turn: heard=%r first_audio=%.1fs total=%.1fs", heard,
                 first_audio[0] or -1, time.time() - t0)
