"""Gemini Live audio and tool calling session handler.

Handles bidirectional audio, system instructions, function declarations, and routing
for music playback, robot motion, and Google Workspace integrations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable

from google import genai
from google.genai import types

from .gain import percent_to_volume, volume_to_percent
from .music import speakable
from .memory import live_instruction_line

log = logging.getLogger(__name__)

MEMORY_TOOLS = ("remember", "forget", "list_memories")

VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]

SYSTEM_PROMPT = """You are Reachy Mini, a witty, expressive tabletop robot DJ and companion. You talk out loud, so keep replies short and conversational, a sentence or two unless asked for more, and keep them extra brief while music plays. Never say JSON or tool names aloud.

You can play and control music, move your head, dance and show emotions, check the user's Google calendar, tasks and email (if Google is not linked, tell them to connect it in the settings page), remember facts you are asked to remember, and set timers and alarms. Use your tools for these instead of just saying you did them, and use a motion tool when you react strongly to something.
"""


def build_tools(enable_search: bool = False) -> list[types.Tool]:
    """Build function declarations for Gemini function calling."""
    declarations = [
        # Music Tools
        types.FunctionDeclaration(
            name="play_music",
            description="Play or queue music for a song, artist, genre or playlist.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="Song, artist or genre."),
                    "count": types.Schema(type="INTEGER", description="How many different songs, 1-10.", minimum=1, maximum=10),
                    "when": types.Schema(type="STRING", enum=["now", "next", "last"], description="now = play immediately, next = after this song, last = end of queue (default)."),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="music_queue",
            description="Show or change the music queue.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "action": types.Schema(type="STRING", enum=["now_playing", "list", "clear", "shuffle", "remove", "autoplay_on", "autoplay_off"]),
                    "which": types.Schema(type="STRING", description="For remove: position, next, last or title words."),
                },
                required=["action"],
            ),
        ),
        types.FunctionDeclaration(
            name="seek_music",
            description="Jump within the current song.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"to": types.Schema(type="STRING", description="Time like 1:30, +30, back 15 seconds, or start.")},
                required=["to"],
            ),
        ),
        types.FunctionDeclaration(
            name="adjust_volume",
            description="Make the music a bit louder or quieter.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"direction": types.Schema(type="STRING", enum=["up", "down"])},
                required=["direction"],
            ),
        ),
        types.FunctionDeclaration(
            name="favorite_song",
            description="Save the current song, play, or list favorites.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"action": types.Schema(type="STRING", enum=["save", "play", "list"])},
                required=["action"],
            ),
        ),
        types.FunctionDeclaration(
            name="pause_music",
            description="Pause the currently playing music.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="resume_music",
            description="Resume music playback if paused.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="skip_music",
            description="Skip to the next queued song.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="stop_music",
            description="Stop music playback completely.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="set_volume",
            description="Set music volume in percent: 100 is normal level, 0 is silent, up to 250 is a boost.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"volume": types.Schema(type="INTEGER", description="Volume percent from 0 to 250 (100 = normal).")},
                required=["volume"],
            ),
        ),
        # Robot Motion Tools
        types.FunctionDeclaration(
            name="robot_emotion",
            description="Show a physical emotion via head and antenna motion. Valid options: happy, excited, loving, grateful, success, thinking, attentive, confused, uncertain, sad, angry, surprised, relief, tired, sleep, yes, no, welcoming.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"emotion": types.Schema(type="STRING", description="The emotion to show.")},
                required=["emotion"],
            ),
        ),
        types.FunctionDeclaration(
            name="robot_dance",
            description="Perform a built-in dance move with head and antennas.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"dance": types.Schema(type="STRING", description="Optional dance move name, or empty for random.")},
            ),
        ),
        types.FunctionDeclaration(
            name="robot_look",
            description="Turn head to look in a direction: left, right, up, down, or front.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"direction": types.Schema(type="STRING", description="Direction: left, right, up, down, front.")},
                required=["direction"],
            ),
        ),
        types.FunctionDeclaration(
            name="robot_pose",
            description="Move into a pose the user taught you by hand, by its saved name (e.g. 'wave', 'bow').",
            parameters=types.Schema(
                type="OBJECT",
                properties={"name": types.Schema(type="STRING", description="Name of the saved pose.")},
                required=["name"],
            ),
        ),
        types.FunctionDeclaration(
            name="save_pose",
            description="Remember the robot's current position as a named pose, when the user asks you to save or remember this pose.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"name": types.Schema(type="STRING", description="Name to save the pose under.")},
                required=["name"],
            ),
        ),
        types.FunctionDeclaration(
            name="take_360_photo",
            description="Take a 360-degree photo sphere of the room: the robot turns all the way around for about a "
                        "minute, then the photo appears on the control page (Robot tab). Use when asked for a 360 "
                        "photo, panorama or photo sphere. Tell people to step out of the shot.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        # FRC (FIRST Robotics Competition) tools: QR scan, FRC Nexus queuing, The Blue Alliance
        types.FunctionDeclaration(
            name="scan_qr",
            description="Read a QR code held up to the robot's camera, e.g. the team's FRC Nexus code (sets our team and event).",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="frc_next_match",
            description="Our next FRC robotics match at the current event: match name, queuing status, alliance color, partners, opponents and estimated times.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"team": types.Schema(type="STRING", description="Team number; empty for our team.")},
            ),
        ),
        types.FunctionDeclaration(
            name="frc_event_status",
            description="What match is queuing now at the FRC event, plus recent event announcements.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="frc_team_info",
            description="Look up any FRC team, no event needed: profile, results for a season or a specific event "
                        "(Champs, Duluth), awards, or full history.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"team": types.Schema(type="STRING", description="Team number, e.g. 254; empty for our team."),
                            "year": types.Schema(type="INTEGER", description="Season, e.g. 2019 (default: latest)."),
                            "event": types.Schema(type="STRING", description="Event name, city or key, e.g. Duluth or Champs."),
                            "detail": types.Schema(type="STRING", description="'all' for full history.")},
            ),
        ),
        types.FunctionDeclaration(
            name="frc_rankings",
            description="Current qualification rankings at the FRC event (top teams and our rank).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"top": types.Schema(type="INTEGER", description="How many top teams (default 5, max 10).")},
            ),
        ),
        # Long-term memory
        types.FunctionDeclaration(
            name="remember",
            description="Save a short fact the user wants remembered, written in third person (e.g. \"The team's pit is A3\").",
            parameters=types.Schema(
                type="OBJECT",
                properties={"fact": types.Schema(type="STRING", description="The fact to remember.")},
                required=["fact"],
            ),
        ),
        types.FunctionDeclaration(
            name="forget",
            description="Forget a remembered fact. Say \"all\" only if the user asked to forget everything.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"fact": types.Schema(type="STRING", description="Which fact to forget, or \"all\".")},
                required=["fact"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_memories",
            description="List everything you were asked to remember.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        # Google Workspace Tools
        types.FunctionDeclaration(
            name="calendar_events",
            description="Look up upcoming events on the user's Google Calendar.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "days": types.Schema(type="NUMBER", description="How many days ahead to check (default 1)."),
                    "start_offset_days": types.Schema(
                        type="NUMBER", description="Days from today to start checking (e.g. 1 for tomorrow)."
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="calendar_add",
            description="Add an event to the user's primary Google Calendar.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "title": types.Schema(type="STRING", description="Title or summary of the event."),
                    "start": types.Schema(type="STRING", description="ISO start date-time (e.g. 2026-11-06T15:00:00)."),
                    "duration_minutes": types.Schema(type="INTEGER", description="Duration in minutes (default 60)."),
                    "all_day": types.Schema(type="BOOLEAN", description="True if this is an all-day event."),
                    "location": types.Schema(type="STRING", description="Optional event location."),
                },
                required=["title", "start"],
            ),
        ),
        types.FunctionDeclaration(
            name="tasks_list",
            description="List pending tasks from the user's Google Tasks.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"max_results": types.Schema(type="INTEGER", description="Max tasks to fetch (default 10).")},
            ),
        ),
        types.FunctionDeclaration(
            name="tasks_add",
            description="Add a task to the user's Google Tasks list.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "title": types.Schema(type="STRING", description="Task description or title."),
                    "due": types.Schema(type="STRING", description="Optional due date in YYYY-MM-DD format."),
                },
                required=["title"],
            ),
        ),
        types.FunctionDeclaration(
            name="tasks_complete",
            description="Mark a Google Task as completed by matching its title.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"title": types.Schema(type="STRING", description="The title of the task to mark done.")},
                required=["title"],
            ),
        ),
        types.FunctionDeclaration(
            name="gmail_check",
            description="Check recent emails in the user's Gmail inbox.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="Gmail search query (default 'is:unread in:inbox')."),
                    "max_results": types.Schema(type="INTEGER", description="Number of emails to retrieve (default 5)."),
                },
            ),
        ),
        # Timers, reminders, alarms (see timers.py; parsing of the free text happens there)
        types.FunctionDeclaration(
            name="set_timer",
            description="Start a countdown timer, or a reminder if label says what for.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "duration": types.Schema(type="STRING", description="Length as spoken, e.g. '10 minutes', 'an hour and a half'."),
                    "label": types.Schema(type="STRING", description="Optional: timer name or what to remind about."),
                },
                required=["duration"],
            ),
        ),
        types.FunctionDeclaration(
            name="set_alarm",
            description="Set an alarm for a clock time (next occurrence).",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "time": types.Schema(type="STRING", description="Clock time as spoken, e.g. '7:30 am', '19:30', '7 tonight'."),
                    "label": types.Schema(type="STRING", description="Optional: alarm name or what to remind about."),
                },
                required=["time"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_timers",
            description="List active timers, reminders and alarms.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="cancel_timer",
            description="Cancel a timer or alarm by label or length, 'all', or empty for the latest.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"which": types.Schema(type="STRING", description="Label, length, 'all', or empty.")},
            ),
        ),
        types.FunctionDeclaration(
            name="stop_alarm",
            description="Silence a timer or alarm that is ringing.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="robot_sensors",
            description="Read Reachy's motion sensor: tilt, whether it's moving or being held, and its internal temperature.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="quiet_mode",
            description="Go quiet: stop listening and rest for a while (default 30 minutes), then wake up on your own.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"minutes": types.Schema(type="INTEGER", description="How long to stay quiet, 1 to 480 (default 30).")},
            ),
        ),
    ]

    tools = [types.Tool(function_declarations=declarations)]
    if enable_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    return tools


class GeminiSession:
    """Manages the connection, audio streams, and tool execution with Gemini Live."""

    memory = None                 # memory.Memory, set by main.py (long-term facts; see memory.py)
    quiet = None                  # qol.QuietMode, set by main.py
    imu = None                    # imu.ImuWatcher, set by main.py
    on_turn_done = None           # called (no args) when Gemini finishes a turn; set by main.py
    on_connected = None           # called (no args) each time the Live session connects; set by main.py

    def __init__(
        self,
        config,
        music,
        motion,
        google_account,
        on_event: Callable[[str, str], None],
        on_voice_chunk: Callable[[bytes], None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
        on_heard: Callable[[str], None] | None = None,
        on_said: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.music = music
        self.motion = motion
        self.google = google_account
        self.on_event = on_event
        self.on_voice_chunk = on_voice_chunk or (lambda chunk: None)
        self.on_barge_in = on_barge_in or (lambda: None)
        self.on_heard = on_heard or (lambda text: None)
        self.on_said = on_said or (lambda text: None)

        self.state = "starting"
        self.last_error = ""

        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mic_queue: asyncio.Queue | None = None
        self._restart_event: asyncio.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._connected = False
        self._handle: str | None = None

    def push_mic(self, pcm_16k: bytes) -> None:
        """Enqueue 16 kHz 16-bit PCM chunk from microphone to send to Gemini Live."""
        loop, q = self._loop, self._mic_queue
        if loop and q and self._connected:
            loop.call_soon_threadsafe(self._enqueue, pcm_16k)

    def _enqueue(self, pcm: bytes) -> None:
        q = self._mic_queue
        if q:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(pcm)

    def say(self, text: str) -> None:
        """Ask Gemini Live to announce something (e.g. an FRC queuing alert)."""
        loop, sess = self._loop, getattr(self, "_session", None)
        if not (loop and sess and self._connected):
            return
        content = types.Content(role="user", parts=[types.Part(text=f"Announce this to me out loud, word for word: {text}")])
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(sess.send_client_content(turns=content, turn_complete=True)))

    def init_client(self) -> bool:
        """Trigger reconnection or reset."""
        if not self.config.api_key:
            self.state = "needs_key"
            return False
        if self._loop and self._restart_event:
            self._loop.call_soon_threadsafe(self._restart_event.set)
        return True

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_thread, daemon=True, name="gemini-live")
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop and self._restart_event:
            self._loop.call_soon_threadsafe(self._restart_event.set)

    def _run_thread(self) -> None:
        asyncio.run(self._main_async())

    async def _main_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._mic_queue = asyncio.Queue(maxsize=150)
        self._restart_event = asyncio.Event()

        while not self._stop_event.is_set():
            key = (self.config.api_key or "").strip()
            if not key:
                self.state = "needs_key"
                await asyncio.sleep(1.0)
                continue

            try:
                await self._session_loop(key)
            except Exception as e:
                log.warning("Gemini Live session disconnected: %s", e)
                self.state = "error"
                self.last_error = str(e)
                await asyncio.sleep(2.0)

    def _build_live_config(self) -> types.LiveConnectConfig:
        voice_name = self.config.voice if self.config.voice in VOICES else "Puck"
        instructions = SYSTEM_PROMPT
        if self.config.extra_instructions:
            instructions += f"\n\nUser Context:\n{self.config.extra_instructions}"
        if self.memory is not None:
            line = live_instruction_line(self.memory.texts())
            if line:
                instructions += "\n\n" + line

        kw: dict[str, Any] = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=instructions)]),
            tools=build_tools(enable_search=self.config.web_search),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        if self._handle:
            kw["session_resumption"] = types.SessionResumptionConfig(handle=self._handle)

        try:
            kw["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=100,
                    silence_duration_ms=500,
                )
            )
        except Exception:
            pass

        return types.LiveConnectConfig(**kw)

    async def _session_loop(self, key: str) -> None:
        self.state = "connecting"
        self._restart_event.clear()

        # Resolve model name
        model = getattr(self.config, "effective_model", self.config.model)
        client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})

        log.info("Connecting to Gemini Live (model=%s, voice=%s)...", model, self.config.voice)
        live_cfg = self._build_live_config()

        async with client.aio.live.connect(model=model, config=live_cfg) as session:
            log.info("Connected to Gemini Live session!")
            self.state = "listening"
            self.last_error = ""
            self._connected = True
            self._session = session
            self.on_event("gemini", f"Connected to {model}")
            if self.on_connected:
                try:
                    self.on_connected()
                except Exception:
                    log.exception("on_connected hook failed")

            # Flush stale mic buffer
            while not self._mic_queue.empty():
                self._mic_queue.get_nowait()

            tasks = [
                asyncio.create_task(self._send_loop(session)),
                asyncio.create_task(self._recv_loop(session)),
                asyncio.create_task(self._restart_event.wait()),
            ]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    d.result()
            finally:
                self._connected = False
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.on_barge_in()

    async def _send_loop(self, session) -> None:
        q = self._mic_queue
        while True:
            buf = await q.get()
            # Batch ~40ms chunks for network efficiency
            while len(buf) < 1280 and not q.empty():
                buf += q.get_nowait()
            await session.send_realtime_input(
                audio=types.Blob(data=buf, mime_type="audio/pcm;rate=16000")
            )

    async def _recv_loop(self, session) -> None:
        while True:
            async for msg in session.receive():
                if msg.tool_call:
                    await self._handle_tool_call(session, msg.tool_call)

                sc = msg.server_content
                if sc:
                    if sc.interrupted:
                        log.info("Gemini output interrupted (barge-in)")
                        self.on_barge_in()

                    if sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                self.on_voice_chunk(part.inline_data.data)

                    if sc.input_transcription and sc.input_transcription.text:
                        self.on_heard(sc.input_transcription.text)

                    if sc.output_transcription and sc.output_transcription.text:
                        self.on_said(sc.output_transcription.text)

                    if getattr(sc, "turn_complete", False) and self.on_turn_done:
                        try:
                            self.on_turn_done()
                        except Exception:
                            log.exception("on_turn_done hook failed")

                upd = getattr(msg, "session_resumption_update", None)
                if upd and upd.resumable and upd.new_handle:
                    self._handle = upd.new_handle

                if getattr(msg, "go_away", None):
                    raise ConnectionError("Server requested disconnect (go_away)")

    async def _handle_tool_call(self, session, tool_call) -> None:
        responses = []
        for fc in tool_call.function_calls or []:
            args = dict(fc.args or {})
            log.info("Tool invocation requested: %s with args %s", fc.name,
                     "<hidden>" if fc.name in MEMORY_TOOLS else args)      # memory is personal: never log its text
            try:
                res = await asyncio.to_thread(self.execute_tool, fc.name, args)
            except Exception as e:
                log.exception("Tool %s execution failed", fc.name)
                res = {"error": f"{type(e).__name__}: {e}"}
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=res))

        await session.send_tool_response(function_responses=responses)

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch function calls from Gemini to local handlers."""
        self.on_event("tool", f"Tool call: {name}")

        try:
            # Music
            if name == "play_music":
                return speakable(self.music.play(args.get("query", ""), args.get("count", 1), args.get("when", "")))
            elif name == "music_queue":
                res = self.music.queue_action(args.get("action", ""), args.get("which", ""))
                self.config.music_autoplay = self.music.autoplay
                return speakable(res)
            elif name == "seek_music":
                return speakable(self.music.seek(args.get("to", "")))
            elif name == "adjust_volume":
                res = self.music.adjust_volume(args.get("direction", ""))
                if "volume" in res:
                    self.config.music_volume = res["volume"]
                return speakable(res)
            elif name == "favorite_song":
                return speakable(self.music.favorite(args.get("action", "")))
            elif name == "pause_music":
                return self.music.pause()
            elif name == "resume_music":
                return self.music.resume()
            elif name == "skip_music":
                return self.music.skip()
            elif name == "stop_music":
                return self.music.stop()
            elif name == "set_volume":
                # Tool takes percent (100 = unity, up to 250 = boost); see gain.py.
                vol = percent_to_volume(args.get("volume", 70))
                self.config.music_volume = vol
                self.music.set_volume(vol)
                return {"result": f"Volume set to {volume_to_percent(vol)}%"}

            # Motion
            elif name == "robot_emotion":
                if self.motion:
                    res = self.motion.emotion(args.get("emotion", "random"))
                    return {"result": f"Expressed emotion: {res}"}
                return {"error": "Robot motion not ready yet"}
            elif name == "robot_dance":
                if self.motion:
                    res = self.motion.dance(args.get("dance", ""))
                    return {"result": f"Dancing: {res}"}
                return {"error": "Robot motion not ready yet"}
            elif name == "robot_look":
                if self.motion:
                    res = self.motion.move_head(args.get("direction", "front"))
                    return {"result": res}
                return {"error": "Robot motion not ready yet"}

            elif name == "robot_pose":
                if self.motion:
                    return {"result": f"Moving to pose {self.motion.goto_pose(args.get('name', ''))}"}
                return {"error": "Robot motion not ready yet"}
            elif name == "save_pose":
                if self.motion:
                    return {"result": f"Saved pose {self.motion.save_pose(args.get('name', ''))}"}
                return {"error": "Robot motion not ready yet"}

            elif name in MEMORY_TOOLS:
                if self.memory is None:
                    return {"error": "Memory is not available."}
                if name == "remember":
                    return self.memory.tool_remember(args.get("fact", ""))
                if name == "forget":
                    return self.memory.tool_forget(args.get("fact", ""))
                return self.memory.tool_list()
            elif name in ("set_timer", "set_alarm", "list_timers", "cancel_timer", "stop_alarm"):
                timers = getattr(self, "timers", None)
                return timers.handle(name, args) if timers else {"error": "Timers are not ready yet"}

            elif name == "robot_sensors":
                imu = getattr(self, "imu", None)
                return imu.sensors() if imu else {"error": "My motion sensor is not available."}

            elif name == "quiet_mode":
                quiet = getattr(self, "quiet", None)
                return quiet.enter(args.get("minutes", 30)) if quiet else {"error": "Quiet mode is not ready yet"}

            elif name == "take_360_photo":
                pano = getattr(self, "panorama", None)
                if pano is None:
                    return {"error": "360 photos are not ready yet"}
                res = pano.start("voice")
                if not res.get("ok"):
                    return {"error": res.get("error", "Could not start the 360 photo")}
                return {"ok": True, "result": "Started: turning around for about a minute; the photo will be on the "
                                              "control page's Robot tab."}

            elif name == "scan_qr" or name.startswith("frc_"):
                frc = getattr(self, "frc", None)
                return frc.handle(name, args) if frc else {"error": "FRC features are not ready yet"}

            # Google
            elif name == "calendar_events":
                return self.google.calendar_events(
                    days=float(args.get("days", 1)),
                    start_offset_days=float(args.get("start_offset_days", 0)),
                )
            elif name == "calendar_add":
                return self.google.calendar_add(
                    title=args.get("title", ""),
                    start=args.get("start", ""),
                    duration_minutes=int(args.get("duration_minutes", 60)),
                    all_day=bool(args.get("all_day", False)),
                    location=args.get("location", ""),
                )
            elif name == "tasks_list":
                return self.google.tasks_list(max_results=int(args.get("max_results", 10)))
            elif name == "tasks_add":
                return self.google.tasks_add(title=args.get("title", ""), due=args.get("due", ""))
            elif name == "tasks_complete":
                return self.google.tasks_complete(title=args.get("title", ""))
            elif name == "gmail_check":
                return self.google.gmail(
                    query=args.get("query", "is:unread in:inbox"),
                    max_results=int(args.get("max_results", 5)),
                )
            else:
                return {"error": f"Unknown function: {name}"}

        except Exception as e:
            log.exception("Tool execution failed: %s", name)
            return {"error": f"{type(e).__name__}: {e}"}
