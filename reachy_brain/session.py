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
_MUSIC_INTENT = re.compile(r"\b(play|queue|cue|put on|spin|throw on)\b(?!.*\b(game|role|trick|joke|part)\b)", re.I)


# "how hot are you", "are you level", "what does your IMU say": the robot reads its own sensor (robot_sensors).
# Needs "you/your" so "what's the temperature outside / in Chicago / today" still goes to get_weather.
_SENSOR_INTENT = re.compile(
    r"\bhow (hot|warm) are you\b|\bare you (hot|warm|overheating|running hot)\b|"
    r"\byour (internal |core |body |current )?temp(erature)?\b(?!.{0,25}\b(outside|today|tomorrow|tonight|there|weather)\b)|"
    r"\bare you (level|tilted|tipped( over)?|leaning|upright|straight|upside down|on your side|lying down)\b|"
    r"\bare you being (picked up|held|carried|moved|shaken|lifted)\b|\bwhich way is up\b|"
    r"\b(imu|gyro(scope)?|accelerometer)\b|\bi[ .]m[ .]u\b|\b(motion|your) sensors?\b.{0,15}\b(say|read|reading|show|tell)\b|"
    r"\b(read|check)\b.{0,12}\b(motion |your |the )?sensors?\b", re.I)


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
    # the robot's own motion sensor / temperature ("your", "are you ..."); outdoor temperature questions stay weather
    (_SENSOR_INTENT, "robot_sensors"),
    (re.compile(r"\b(weather|temperature|forecast|raining|snowing|how (hot|cold|warm) is it)\b", re.I), "get_weather"),
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


_FRC_CONTEXT = re.compile(r"\b(team|frc)\s*\d{1,5}\b|frc_team_info|frc_rankings|frc_next_match", re.I)
# follow-ups about a team already being discussed ("how did they do at champs", "their 2026 season", ...)
_FRC_FOLLOW = re.compile(
    r"\b(how (did|do|are|have) (they|we)\b|(their|our) (\d{4} )?(season|competitions?|events?|records?|history|performance|results?|"
    r"stats)\b|(their|our) \w+ (competition|event|regional)\b|all (of )?(their|our) (info|information|history|stats|data)|"
    r"(their|our) \d{4}\b|at (the )?(champs|champions|worlds|championship|regional|district|state)\b|"
    r"more (info|information) (on|about) (their|our|them|that team)\b)", re.I)


def forced_tool(heard: str, available: set[str], recent: str = "") -> str | None:
    """Name of the tool this request must start with, or None to let the model decide. `recent` is the text of the
    last few turns: it lets a follow-up with no team number ("how did they do at champs") reach frc_team_info."""
    if "play_music" in available and _PLAY_ARGS.search(_norm(heard)) and not _PLAY_NOT.search(heard):
        return "play_music"
    if "frc_team_info" in available and _FRC_FOLLOW.search(heard) and _FRC_CONTEXT.search(recent):
        return "frc_team_info"
    for rx, name in _INTENTS:
        if rx.search(heard) and name in available:
            return name
    if _MUSIC_INTENT.search(_norm(heard)) and "play_music" in available:
        return "play_music"
    return None


# What Whisper "hears" in silence or room noise; never a real request.
GHOST_TRANSCRIPTS = {"you", "thank you", "thanks", "thanks for watching", "bye", "bye bye", "the end"}


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

_NUMWORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
             "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
_N10 = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"


def _norm(heard: str) -> str:
    t = heard.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9.:' ]", " ", t)
    t = " ".join(t.split()).rstrip(". :")
    # Whisper writes "queue" as "cue" / "Q" ("please cue 10 minecraft songs", "add it to the q")
    t = re.sub(r"^((?:please |can you |could you |hey reachy )*)(?:cue|q|kew)\b", r"\1queue", t)
    return re.sub(r"\bto (the|my) (?:cue|q|kew)\b", r"to \1 queue", t)


def _num(word: str) -> int | None:
    w = word.strip().lower()
    return int(w) if w.isdigit() else _NUMWORDS.get(w)


# Music transport / queue / volume / favorites shortcuts: (tool, fallback ack, args, whole-utterance pattern).
# They only fire when the robot advertises the tool; the robot's {"say": ...} is spoken in preference to the ack.
_QUICK = [
    ("pause_music", "Paused.", {}, re.compile(r"^(please |can you |could you )*(pause|hold)( the| this| that)?( music| song| track| it)?( please)?$")),
    ("resume_music", "Back to the music.", {}, re.compile(
        r"^(please |can you |could you )*(resume|unpause|continue|keep going|play)( the| this| that)?"
        r"( music| song| track| it)?( again| back)?( please)?$")),
    ("skip_music", "Skipping.", {}, re.compile(
        r"^(please |can you |could you )*(skip|next|play( the)? next|skip to( the)? next)( the| this| that)?( music| song| track| one)?( please)?$")),
    ("stop_music", "Music stopped.", {}, re.compile(r"^(please |can you |could you )*stop( the| this| that)?( music| song| track| playing)( please)?$")),
    ("music_queue", "Let me check.", {"action": "now_playing"}, re.compile(
        rf"^{_PLEASE}(?:what'?s|what is|whats) (?:currently |now )?playing(?: right now| now)?$|"
        rf"^{_PLEASE}(?:what'?s|what is|whats) (?:this|that)(?: song| track)?(?: playing)?$|"
        rf"^{_PLEASE}what (?:song|track) is (?:this|that|playing)(?: right now| now)?$|"
        rf"^{_PLEASE}who (?:sings|is singing|plays|is playing) (?:this|that)(?: song| track)?$|"
        rf"^{_PLEASE}who'?s singing(?: this| that)?$")),
    ("music_queue", "Let me check.", {"action": "list"}, re.compile(
        rf"^{_PLEASE}(?:what'?s|what is|whats) (?:up )?next(?: in the queue| in line)?$|"
        rf"^{_PLEASE}(?:what'?s|what is|whats) (?:in|on) the (?:queue|playlist)$|"
        rf"^{_PLEASE}(?:what'?s|what is) coming up(?: next)?$|"
        rf"^{_PLEASE}(?:show|list|read)(?: me| out)?(?: the| my) (?:queue|playlist)$")),
    ("music_queue", "Queue cleared.", {"action": "clear"}, re.compile(
        rf"^{_PLEASE}(?:clear|empty)(?: out)?(?: the| my)? (?:queue|playlist)$")),
    ("music_queue", "Shuffled.", {"action": "shuffle"}, re.compile(
        rf"^{_PLEASE}shuffle(?: up)? (?:the queue|the music|it|my music|the playlist|the songs)(?: please)?$")),
    ("music_queue", "Okay, I'll keep the music going.", {"action": "autoplay_on"}, re.compile(
        rf"^{_PLEASE}(?:keep the music (?:going|playing)|keep it (?:going|playing)|keep playing (?:music|songs)|"
        rf"(?:turn on|enable|start) autoplay)$")),
    ("music_queue", "Okay, I'll stop after this song.", {"action": "autoplay_off"}, re.compile(
        rf"^{_PLEASE}(?:stop after this (?:song|track|one)|(?:turn off|disable) autoplay|don'?t keep the music going)$")),
    ("adjust_volume", "Louder.", {"direction": "up"}, re.compile(
        rf"^{_PLEASE}(?:make it |a (?:bit|little) |bit |little )?louder(?: please)?$|"
        rf"^{_PLEASE}turn (?:it|that|the (?:music|volume)) up(?: a bit| a little)?(?: please)?$|"
        rf"^{_PLEASE}turn up the (?:volume|music)(?: a bit| a little)?(?: please)?$|"
        rf"^{_PLEASE}(?:volume up|raise the volume|crank it up)(?: please)?$")),
    ("adjust_volume", "Quieter.", {"direction": "down"}, re.compile(
        rf"^{_PLEASE}(?:make it |a (?:bit|little) |bit |little )?(?:quieter|softer)(?: please)?$|"
        rf"^{_PLEASE}turn (?:it|that|the (?:music|volume)) down(?: a bit| a little)?(?: please)?$|"
        rf"^{_PLEASE}turn down the (?:volume|music)(?: a bit| a little)?(?: please)?$|"
        rf"^{_PLEASE}(?:volume down|lower the volume|not so loud)(?: please)?$")),
    ("favorite_song", "Saved to your favorites.", {"action": "save"}, re.compile(
        rf"^{_PLEASE}(?:save (?:this|that)(?: song| track| one)?|i (?:like|love) (?:this|that)(?: song| track| one)?|"
        rf"add (?:this|that)(?: song| track| one)? to (?:my )?favou?rites?(?: songs| list)?|"
        rf"favou?rite (?:this|that)(?: song| track| one)?)$")),
    ("favorite_song", "Playing your favorites.", {"action": "play"}, re.compile(
        rf"^{_PLEASE}play (?:my |our )?(?:favou?rites|favou?rite (?:songs|music|tracks))$")),
    ("robot_sensors", "Let me check.", {}, re.compile(
        rf"^{_PLEASE}(?:how (?:hot|warm) are you(?: right now| now)?|"
        rf"(?:what'?s|what is|whats) your (?:internal |current |core )?temperature(?: right now| now)?|"
        rf"(?:read|check)(?: out)? your (?:imu|sensors?|motion sensor)|"
        rf"what (?:does|do) your (?:imu|sensors?|motion sensor) (?:say|read|show))(?: please)?$")),
]

# "go to sleep" -> quiet_mode: the robot mutes its mic until the talk button / page / timeout wakes it
_QUIET_FOR = rf" for (?P<n>half an|{_NUM})(?: (?P<u>{_UNIT}))?"
_QUICK_QUIET = [
    re.compile(rf"^{_PLEASE}(?:go to sleep|go to bed|stop listening|take a break|go quiet)(?:{_QUIET_FOR})?(?: for now)?$"),
    re.compile(rf"^{_PLEASE}(?:be quiet|be silent|quiet down|sleep|take a nap){_QUIET_FOR}$"),
]


def _quiet_minutes(m: re.Match) -> int:
    n, u = m.groupdict().get("n"), m.groupdict().get("u")
    if not n:
        return 30
    if n == "half an":
        return 30                              # "half an hour"
    val = _num(n)
    if val is None:
        try:
            val = float(n)
        except ValueError:
            return 30
    hours = (u or "minutes").lower().startswith(("hour", "hr"))
    return max(1, min(int(round(val * (60 if hours else 1))), 24 * 60))


# ---------- seek_music: "skip to 1:30", "rewind 10 seconds", "restart the song" ----------
_SECS = r"(?:seconds?|secs?)"
_MINS = r"(?:minutes?|mins?)"
_SEEK_VERB = r"(?:skip|jump|go|seek|fast forward|move|fast forward to)"
_SEEK_FWD = re.compile(rf"^{_PLEASE}(?:skip|jump|go|fast forward|move)? ?(?:ahead|forward)(?: (?P<n>{_NUM}) (?P<u>{_SECS}|{_MINS}))?(?: a bit| a little)?$|"
                       rf"^{_PLEASE}fast forward(?: (?P<n2>{_NUM}) (?P<u2>{_SECS}|{_MINS}))?(?: a bit| a little)?$")
_SEEK_BACK = re.compile(rf"^{_PLEASE}(?:go back|rewind|back up|jump back|skip back|go backwards?|jump backwards?|skip backwards?)"
                        rf"(?: (?P<n>{_NUM}) (?P<u>{_SECS}|{_MINS}))?(?: a bit| a little)?$")
_SEEK_START = re.compile(rf"^{_PLEASE}(?:(?:start|play)(?: the| this| that)? (?:song|track|music|it) over|"
                         rf"restart(?: the| this| that)?(?: song| track| music| it)?|"
                         rf"(?:play |start |take it |go |skip |jump |seek )?(?:from|to) the (?:beginning|start|top)(?: of the (?:song|track))?)$")
_SEEK_ABS = re.compile(rf"^{_PLEASE}{_SEEK_VERB} (?:ahead |forward )?to (?:the )?(?P<t>.+)$")
_SEEK_WORD_MIN = r"one|two|three|four|five|six|seven|eight|nine|ten"
_SEEK_WORD_SEC = r"fifteen|twenty|thirty|forty five|forty|fifty|ten|five"


def _amount(n: str | None, u: str | None, default: int = 10) -> int:
    if not n:
        return default
    v = _num(n)
    if v is None:
        try:
            v = float(n)
        except ValueError:
            return default
    return max(1, int(round(v * (60 if u and u.startswith("min") else 1))))


def _seek_time(x: str) -> str | None:
    """Time text for seek_music from what follows "skip to": 1:30 / 1.30 / 1 30 / one thirty / 2 minutes / the start."""
    x = x.strip()
    if re.fullmatch(r"(?:the )?(?:beginning|start|top)(?: of the (?:song|track))?", x):
        return "start"
    if m := re.fullmatch(r"(\d{1,2})[:. ](\d{2})", x):
        return f"{int(m.group(1))}:{m.group(2)}" if int(m.group(2)) < 60 else None
    if m := re.fullmatch(rf"({_SEEK_WORD_MIN}) (oh )?({_SEEK_WORD_SEC}|{_SEEK_WORD_MIN})", x):
        sec = _num(m.group(3))
        if sec is not None and (sec >= 10 or m.group(2)):
            return f"{_num(m.group(1))}:{sec:02d}"
    if m := re.fullmatch(rf"(?P<a>{_NUM}) (?P<au>{_SECS}|{_MINS})(?: and (?P<b>{_NUM}) (?P<bu>{_SECS}))?", x):
        a = _amount(m.group("a"), m.group("au"), 0)
        if not a:
            return None
        if m.group("au").startswith("min"):
            n = _amount(m.group("a"), "seconds", 0)
            out = f"{n} minute{'s' if n != 1 else ''}"
            return out + (f" {_amount(m.group('b'), 'seconds', 0)} seconds" if m.group("b") else "")
        return f"{a} seconds"
    return None


def seek_command(t: str) -> str | None:
    """`to` text for seek_music when the normalized utterance is a bare seek request, else None."""
    if m := _SEEK_FWD.match(t):
        return f"+{_amount(m.group('n') or m.group('n2'), m.group('u') or m.group('u2'))}"
    if m := _SEEK_BACK.match(t):
        return f"-{_amount(m.group('n'), m.group('u'))}"
    if _SEEK_START.match(t):
        return "start"
    if m := _SEEK_ABS.match(t):
        return _seek_time(m.group("t"))
    return None


# Words that are not a song/artist: "play it now" / "play music next" are left to the model.
_GENERIC_Q = {"it", "this", "that", "the", "a", "some", "music", "some music", "something", "a song", "some songs",
              "the music", "the song", "the next", "the next song", "songs", "a track", "something else", "one"}


def quick_command(heard: str) -> tuple[str, str, dict] | None:
    """('tool_name', spoken ack, args) when the whole utterance is a bare music / timer / quiet command, else None."""
    t = _norm(heard)
    for name, ack, args, rx in _QUICK:
        if rx.match(t):
            return name, ack, dict(args)
    for rx in _QUICK_TIMER:
        if m := rx.match(t):
            return "set_timer", "Done.", {"duration": m.group("dur")}
    if m := _QUICK_CANCEL.match(t):
        return "cancel_timer", "Done.", {"which": "all" if m.group("t") == "timers" else ""}
    if _QUICK_STOP_ALARM.match(t):
        return "stop_alarm", "Done.", {}
    if (to := seek_command(t)) is not None:
        return "seek_music", "Okay.", {"to": to}
    for rx in _QUICK_QUIET:
        if m := rx.match(t):
            mins = _quiet_minutes(m)
            return "quiet_mode", f"Okay, quiet for {mins} minutes.", {"minutes": mins}
    # play_music with an argument we can read without the model
    if m := re.match(rf"^{_PLEASE}play (?P<q>.+?) (?P<w>next|(?:right )?now)$", t):
        q = m.group("q")
        if q not in _GENERIC_Q and not q.endswith(" right") and not re.search(r"\b(game|role|trick|joke|part)\b", q):
            return "play_music", "Okay, queuing that up.", {"query": q, "when": "next" if m.group("w") == "next" else "now"}
    for rx in (rf"^{_PLEASE}(?:queue up|queue|play|add) (?P<n>{_N10}) (?:songs|tracks) by (?P<q>.+)$",
               rf"^{_PLEASE}(?:queue up|queue|play|add) (?P<n>{_N10}) (?P<q>(?!songs|tracks)[a-z0-9' ]+?) (?:songs|tracks)$"):
        if m := re.match(rx, t):
            n = _num(m.group("n"))
            if n and 1 <= n <= 10 and m.group("q") not in _GENERIC_Q:
                # keep "songs" in the search: "minecraft" alone finds gameplay videos, "minecraft songs" finds music
                q = m.group("q") if " by " in rx else m.group("q") + " songs"
                return "play_music", "Okay, queuing that up.", {"query": q, "count": n}
    return None


# ---------- brain-local answers (no robot tool, no LLM) ----------
_LOCAL_TIME = re.compile(rf"^{_PLEASE}(?:what'?s the time|what is the time|what time is it|tell me the time|"
                         rf"do you have the time)(?: now| right now| please)?$")
_LOCAL_DATE = re.compile(rf"^{_PLEASE}(?:what'?s|what is) (?:the|today'?s) date(?: today)?$|"
                         rf"^{_PLEASE}what day is (?:it|today)(?: today)?$")
_LOCAL_REPEAT = re.compile(rf"^{_PLEASE}(?:say that again|say it again|repeat that|repeat it|what did you say|come again|"
                           rf"say again|pardon|repeat yourself)(?: please)?$")
_LOCAL_CANCEL = re.compile(r"^(?:never ?mind|cancel|cancel that|forget it|nothing|scratch that|ignore that)$")
_LOCAL_HELP = re.compile(rf"^{_PLEASE}(?:what can you do|help|help me|what are your features|what do you do|"
                         rf"what are your capabilities|what can i ask you|what can i say)(?: please)?$")

_HELP_AREAS = [
    ("music", {"play_music", "music_queue"}),
    ("moves and emotions", {"robot_emotion", "robot_dance", "robot_pose", "robot_look"}),
    ("camera", {"look"}),
    ("weather and web", {"get_weather", "web_search"}),
    ("timers and alarms", {"set_timer", "set_alarm"}),
    ("memory", {"remember"}),
    ("FRC", {"frc_next_match", "frc_team_info", "frc_rankings"}),
]


def help_text(tools: set[str]) -> str:
    areas = [label for label, names in _HELP_AREAS if names & tools]
    if not areas:
        return "I can chat and answer questions. Just ask."
    joined = areas[0] if len(areas) == 1 else ", ".join(areas[:-1]) + (", and " if len(areas) > 2 else " and ") + areas[-1]
    return f"I can help with {joined}. Just ask!"


def local_answer(heard: str, tools: set[str], last_said: str = "",
                 now: datetime.datetime | None = None) -> tuple[str, bool] | None:
    """(text to speak, is_it_a_real_answer) for requests the brain answers itself, else None. The flag is False for
    "Okay." / repeat / "nothing said yet" replies, which must not replace what "say that again" repeats."""
    t = _norm(heard)
    now = now or datetime.datetime.now()
    if _LOCAL_CANCEL.match(t):
        return "Okay.", False
    if _LOCAL_REPEAT.match(t):
        return (last_said, False) if last_said else ("I haven't said anything yet.", False)
    if _LOCAL_TIME.match(t):
        return f"It's {now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}.", True
    if _LOCAL_DATE.match(t):
        return f"It's {now.strftime('%A, %B')} {now.day}.", True
    if _LOCAL_HELP.match(t):
        return help_text(tools), True
    return None


# "add X to the queue" / "play X next" / "play 3 songs by X": always a play_music call; the hint names the new args.
_PLAY_ARGS = re.compile(r"\b(add|put)\b.+\bto (the|my) (queue|playlist)\b|\bqueue up\b|\bplay\s+\S.*\s(next|now)$|"
                        rf"\b(play|queue|add)\s+(?:{_N10})\s+(?:\w+\s+){{0,2}}(songs|tracks)\b|"
                        rf"\b(wanted|want|asked for|give me|need)\s+(?:{_N10})\s+(?:\w+\s+){{0,2}}(songs|tracks)\b", re.I)
_PLAY_NOT = re.compile(r"\b(match|matches|queuing|qualification|team|frc|timer|alarm|calendar|game|role|trick|joke|part)\b", re.I)


def play_hint(heard: str) -> str:
    """Extra words for the forced-call hint naming play_music's optional args, e.g. '; use when="next", count=3'."""
    t = _norm(heard)
    parts = []
    if re.search(r"\b(next|up next)$", t):
        parts.append('when="next"')
    elif re.search(r"\bnow$", t):
        parts.append('when="now"')
    elif re.search(r"\badd\b.+\bto (the|my) (queue|playlist)\b|\bqueue up\b", t):
        parts.append('when="last"')
    m = re.search(rf"\b({_N10})\s+(?:\w+\s+){{0,2}}(?:songs|tracks)\b", t)
    if m and (n := _num(m.group(1))) and 1 <= n <= 10:
        parts.append(f"count={n}")
    return ("; use " + ", ".join(parts)) if parts else ""


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
        self._last_said = ""                        # Reachy's last real answer (never a filler), for "say that again"
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
                    self._last_said = audio[1]
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
        # The saved file is named after system prompt + tools only (the slow ~2k-token part). Remembered facts come
        # after them, so a file saved with older facts still covers that prefix: llama-server re-reads only the rest.
        base = hashlib.sha1(json.dumps([self._system(), self._tools()], sort_keys=True).encode()).hexdigest()
        fname = f"warm-{base[:16]}.bin"
        restored = False
        try:
            restored = bool(hasattr(self.llm, "slot_restore") and self.llm.slot_restore(fname))
            if restored:
                log.info("LLM prompt cache restored from disk in %.1fs", time.time() - t0)
        except Exception as e:
            log.info("prompt cache restore failed: %s", e)
        try:
            msgs = [{"role": "system", "content": self._system()}, *self._pinned(), {"role": "user", "content": "hi"}]
            self.llm.chat_stream(msgs, self._tools(), lambda c: None, threading.Event(), max_tokens=1)
            self.llm.warm_key = key
            log.info("LLM warm-up took %.1fs%s", time.time() - t0, " (after restore)" if restored else "")
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
        # Brain-local answers: no robot tool, no LLM, and NOT stored in history. They add nothing the model needs
        # (time/date/help/repeat/cancel are stateless), and storing them would only lengthen the next prompt;
        # "say that again" works off self._last_said instead.
        local = local_answer(heard, {t["function"]["name"] for t in self._tools()}, self._last_said)
        if local is not None:
            text, real = local
            self.send_json({"type": "event", "kind": "tool", "text": "local answer"})
            self._state("speaking")
            self._speak(text)
            if real:
                self._last_said = text
            log.info("turn: heard=%r local total=%.1fs", heard, time.time() - t0)
            return
        quick = quick_command(heard)
        if quick and any(t["function"]["name"] == quick[0] for t in self.robot_tools):
            name, ack, qargs = quick
            self.send_json({"type": "event", "kind": "tool", "text": f"{name} (voice shortcut)"})
            res = self._robot_call(name, qargs, self.cfg.tool_timeout_s)
            say = (say_of(res) or ack) if not _failed(res) else str((res or {}).get("error", "That didn't work."))[:120]
            self._state("speaking")
            self._speak(say)
            self._last_said = say
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
        recent = " ".join(str(m.get("content") or "") + json.dumps(m.get("tool_calls") or "")
                          for t in self.turns[-3:] for m in t)
        force = forced_tool(heard, {t["function"]["name"] for t in tools}, recent)
        user_msg = {"role": "user", "content": heard}
        if force:
            choice = "required"
            hint = play_hint(heard) if force == "play_music" else ""
            if force == "frc_team_info":
                hint = ("" if re.search(r"\b(team|frc)\s*\d", heard, re.I) else "; take the team from the conversation") + \
                       "; set event, year or detail=\"all\" if asked"
            user_msg = {"role": "user", "content": f"{heard}\n(Call {force} first{hint}. Do not answer from memory.)"}
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
        if (final or reply) and final != "(done)":
            self._last_said = final or reply
        turn = msgs[base:]
        turn.append({"role": "assistant", "content": final or reply or "(done)"})
        self._remember(turn)
        log.info("turn: heard=%r first_audio=%.1fs total=%.1fs", heard,
                 first_audio[0] or -1, time.time() - t0)
