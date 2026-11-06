---
title: Reachy Gemini DJ
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
tags:
  - reachy_mini_python_app
---

# Reachy Gemini DJ 🎧🤖

A voice-powered AI DJ and tabletop assistant for the [Pollen Robotics Reachy Mini](https://huggingface.co/docs/reachy_mini/en/index), powered by Google Gemini Live.

## Highlights
- **Continuous Voice Interaction**: Natural voice chat with native Gemini audio streaming—no wake word required.
- **Smart Music DJ**: Music streaming with automatic audio ducking during conversation, responsive playback controls, and volume tuning.
- **Built-in Robot Motion**: Expressive head gestures, emotions, and dances from Pollen's official motion libraries.
- **Google Workspace**: Direct access to your Google Calendar, Google Tasks, and Gmail (read-only) via local PKCE OAuth.
- **Responsive Dashboard UI**: Fully scaling, mobile & desktop-friendly control room at `http://reachy-mini.local:8042`.

---

## 1-Click Installation (Reachy Mini App Store)

Because this Space is tagged with `reachy_mini_python_app`, you can install it directly onto your robot:

1. Open **Reachy Mini Control** in your browser.
2. Go to the **Applications** tab and click **Discover Apps**.
3. Locate **Reachy Gemini DJ** and click **Install**.

---

## Command Line Installation

### Option A: Install via Reachy Mini REST API
From your computer or terminal:
```bash
curl -X POST http://reachy-mini.local:8000/api/apps/install \
  -H "Content-Type: application/json" \
  -d '{"name": "reachy_gemini_dj", "source_kind": "hf_space", "url": "https://huggingface.co/spaces/sillymastermind/reachy_gemini_dj"}'
```

### Option B: Direct Python Install
On the robot (or in simulation):
```bash
uv pip install git+https://huggingface.co/spaces/sillymastermind/reachy_gemini_dj
```

---

## Quick Setup

1. Open `http://reachy-mini.local:8042` in your browser.
2. Enter your [Google AI Studio Gemini API Key](https://aistudio.google.com/apikey).
3. Connect your Google account under the **Google** tab (optional).
4. Start talking: *"Reachy, play some chill lofi beats"*, *"What's on my calendar tomorrow?"*, or *"Show me a dance!"*

---

## Free local voice (brain server)

The robot's CM4 is too small for speech recognition and a language model, so the free voice runs on a
computer on the same network (a laptop or Mac mini). The robot app stays light and streams mic audio to it.

```
Reachy Mini (CM4)                               Brain computer (laptop / Mac mini)
 app: mic, speaker, motion, music, Google  <-WebSocket->  VAD -> Whisper STT -> llama-server LLM -> Pocket TTS
 runs music/motion/calendar tools                         runs web search, weather, camera reading, Gemini help
```

**Backend modes** (Settings, *Voice backend*):
- **Auto** (default): local brain first. If it is unreachable at start or fails 3 times in a row, the app
  switches to Gemini Live (needs an API key) and switches back when the brain is reachable and the robot is idle.
- **Local only** / **Gemini only**.
- **Gemini for hard questions**: with an API key set, the local model can hand hard questions (maths, code,
  long explanations) to Gemini; the answer is still spoken by the local voice. Say "ask Gemini ..." to force it.

### Brain setup
1. Install: `pip install "reachy_gemini_dj[brain] @ git+https://github.com/erikb-lab/reachy_gemini_dj"`
   (or `pip install ".[brain]"` from a clone). `ffmpeg` is not needed on the brain.
2. Get [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) `llama-server`, and a GGUF model with tool
   calling (tested: Groot-19B-A3B Q4_K_M, plus its `mmproj` file for camera questions).
3. Run:
   ```bash
   reachy-brain --llama-server-bin /path/to/llama-server --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf
   ```
   It listens on `ws://0.0.0.0:8765`. Useful flags: `--llm-threads 4`, `--tts-voice alba`,
   `--llama-url http://host:port` to use a llama-server you already run, `--gemini-key` for Gemini help.
4. In the robot's web UI set **Brain URL** to `ws://<brain-computer-ip>:8765`.

On a CPU-only laptop expect roughly 4-5 s to first spoken word for chat, about 13-15 s for music or weather,
and about 25 s when a web search is needed.

---

## FRC robotics helper (Nexus + The Blue Alliance)

For FIRST Robotics Competition events. No Google services are involved.

1. In **Settings, FRC robotics** paste a free [FRC Nexus](https://frc.nexus/api) API key (live queuing) and/or a
   [Blue Alliance](https://www.thebluealliance.com/account) read key (rankings and results).
2. Press **Scan QR code** (or say "scan this QR code") and hold your team's Nexus code up to Reachy's camera.
   The code is decoded on the robot; team and event are filled in. The private team link is stored only on the
   robot and is never spoken, shown, or sent to any model.
3. Ask: "When's our next match?", "What's queuing now?", "Where is team 254's pit?", "Show me the rankings."
   With **Queuing alerts** on, Reachy says out loud when your match is queuing soon, now queuing or on deck
   (it checks Nexus every 30 seconds).

Match data from [FRC Nexus](https://frc.nexus) and [The Blue Alliance](https://www.thebluealliance.com).
