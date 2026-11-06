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

log = logging.getLogger(__name__)

VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]

SYSTEM_PROMPT = """You are Reachy Mini, an expressive, witty tabletop robot DJ and companion.
You speak aloud to the user using audio, so keep your spoken responses concise, punchy, and conversational (1-2 sentences unless details were requested).

Capabilities:
1. DJ / Music: You search, stream, and control music tracks. Duck or pause when interacting.
2. Motion: You can turn your head, perform built-in dances, and express emotions with head and antenna gestures. When expressing a strong mood or reaction, invoke the robot motion tools.
3. Google Workspace: You can check or schedule calendar events, list or complete tasks, and check unread emails when the user asks.

Guidelines:
- Never say JSON or function call names aloud.
- When music plays, keep responses short so the user can enjoy the music.
- If Google is not connected, politely remind the user they can link their account in the settings page.
"""


def build_tools(enable_search: bool = False) -> list[types.Tool]:
    """Build function declarations for Gemini function calling."""
    declarations = [
        # Music Tools
        types.FunctionDeclaration(
            name="play_music",
            description="Search and play a song, genre, artist, or playlist.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="The song, artist, or music genre to play.")},
                required=["query"],
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
    ]

    tools = [types.Tool(function_declarations=declarations)]
    if enable_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    return tools


class GeminiSession:
    """Manages the connection, audio streams, and tool execution with Gemini Live."""

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
            self.on_event("gemini", f"Connected to {model}")

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

                upd = getattr(msg, "session_resumption_update", None)
                if upd and upd.resumable and upd.new_handle:
                    self._handle = upd.new_handle

                if getattr(msg, "go_away", None):
                    raise ConnectionError("Server requested disconnect (go_away)")

    async def _handle_tool_call(self, session, tool_call) -> None:
        responses = []
        for fc in tool_call.function_calls or []:
            args = dict(fc.args or {})
            log.info("Tool invocation requested: %s with args %s", fc.name, args)
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
                return self.music.play(args.get("query", ""))
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
