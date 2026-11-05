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
- **Mobile-Friendly Web Dashboard**: Clean control room at `http://reachy-mini.local:8042` with setup checklists, motion pads, Google OAuth wizard, and sound sliders.

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
