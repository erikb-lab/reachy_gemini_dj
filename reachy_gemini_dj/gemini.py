"""Gemini Live audio and tool calling session handler.

Handles bidirectional audio, system instructions, function declarations, and routing
for music playback, robot motion, and Google Workspace integrations.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import google.genai as genai
from google.genai import types

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
            description="Set music volume (0-100 percent).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"volume": types.Schema(type="INTEGER", description="Volume level from 0 to 100.")},
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
    """Manages the connection, audio streams, and tool execution with Gemini."""

    def __init__(self, config, music, motion, google_account, on_event: Callable[[str, str], None]):
        self.config = config
        self.music = music
        self.motion = motion
        self.google = google_account
        self.on_event = on_event
        self.client: genai.Client | None = None
        self.state = "starting"
        self.last_error = ""

    def init_client(self) -> bool:
        if not self.config.api_key:
            self.state = "needs_key"
            return False
        try:
            self.client = genai.Client(api_key=self.config.api_key, http_options={"api_version": "v1alpha"})
            self.state = "connecting"
            return True
        except Exception as e:
            self.state = "error"
            self.last_error = str(e)
            return False

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch function calls from Gemini to local handlers."""
        log.info("Tool invocation: %s with args %s", name, args)
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
                vol = max(0, min(100, int(args.get("volume", 70)))) / 100.0
                self.config.music_volume = vol
                self.music.set_volume(vol)
                return {"result": f"Volume set to {int(vol * 100)}%"}

            # Motion
            elif name == "robot_emotion":
                res = self.motion.emotion(args.get("emotion", "random"))
                return {"result": f"Expressed emotion: {res}"}
            elif name == "robot_dance":
                res = self.motion.dance(args.get("dance", ""))
                return {"result": f"Dancing: {res}"}
            elif name == "robot_look":
                res = self.motion.move_head(args.get("direction", "front"))
                return {"result": res}

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
