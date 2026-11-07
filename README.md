---
title: Reachy Gemini DJ
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
short_description: Voice DJ and helper for Reachy Mini with a free local voice
tags:
  - reachy_mini_python_app
---

# Reachy Gemini DJ 🎧🤖

A voice DJ and tabletop helper for the [Pollen Robotics Reachy Mini](https://huggingface.co/docs/reachy_mini/en/index).
Talk to it with no wake word. It plays music, moves and emotes, looks through its camera, checks the weather and
the web, and helps FIRST Robotics teams at events.

The voice is **free and local by default**. Speech recognition, the language model and the voice run on a computer
on your network, not in the cloud. **Google Gemini Live** is optional. You can use it as a backup, or for hard
questions.

## What it can do
- **Music DJ.** Say "play some chill lo-fi" or "play Daft Punk". Music is streamed, never downloaded, and drops in
  volume while you talk. "Pause", "play the music", "skip" and "stop the music" work instantly.
- **Moves and emotions.** Emotions, dances and head moves come from Pollen's official libraries. **Learn poses**
  lets you turn the motors off, pose Reachy by hand, lock the pose and save it by name. Then say "go to my
  playing-music pose".
- **Camera.** Ask "what do you see?" and the local vision model describes what is in front of Reachy.
- **Weather and web search.** Both are free (Open-Meteo and DuckDuckGo) and need no keys.
- **FRC helper.** Scan your team's FRC Nexus QR code, hear queuing alerts, and look up any team's history and
  results ([below](#frc-robotics-helper-nexus--the-blue-alliance)).
- **Google Calendar, Tasks and Gmail (optional, read-only for Gmail).** Connect them in the **Google** tab.
- **Controller web page** at `http://reachy-mini.local:8042`. It has a dark mode, live volume sliders for the
  music and for Reachy's voice, and works on phones.

---

## Install on the robot

**App store (easiest):** open **Reachy Mini Control**, then **Applications**, then **Discover Apps**. Find
**Reachy Gemini DJ** and click **Install**.

**REST API:**
```bash
curl -X POST http://reachy-mini.local:8000/api/apps/install \
  -H "Content-Type: application/json" \
  -d '{"name": "reachy_gemini_dj", "source_kind": "hf_space", "url": "https://huggingface.co/spaces/sillymastermind/reachy_gemini_dj"}'
```

**pip (on the robot or in simulation):**
```bash
uv pip install git+https://huggingface.co/spaces/sillymastermind/reachy_gemini_dj
```

## Quick setup
1. Start the **brain** on a computer on the same network ([see below](#the-brain-free-local-voice)).
2. Open `http://reachy-mini.local:8042` and set **Brain URL** in Settings to `ws://<that-computer's-ip>:8765`.
3. Optional: paste a [Gemini API key](https://aistudio.google.com/apikey) to use Gemini as a backup and for hard
   questions.
4. Start talking. Try "play some lo-fi", "show me you're happy", "what do you see?", or "what's the weather
   in Chicago?".

The app sets the speaker to its full hardware level when it starts, because the robot ships at 72%. Use the
music and voice sliders to make it quieter.

---

## The brain (free local voice)

The robot's Raspberry Pi is too small to run speech recognition and a language model. The robot app stays light
and streams microphone audio to a "brain" on a laptop or Mac mini.

```
Reachy Mini (CM4)                                  Brain computer (laptop / Mac mini)
 mic, speaker, motion, music, camera  <-WebSocket->  voice detection -> Whisper -> llama-server -> Pocket TTS
 runs music / motion / FRC / Google tools            runs web search, weather, camera questions, Gemini help
```

### Brain setup
1. Install it with `pip install "reachy_gemini_dj[brain] @ git+https://github.com/erikb-lab/reachy_gemini_dj"`,
   or with `pip install ".[brain]"` from a clone.
2. Get [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) `llama-server` and a GGUF model that supports
   tool calling. It was tested with Groot-19B-A3B Q4_K_M, plus its `mmproj` file for camera questions.
3. Run:
   ```bash
   reachy-brain --llama-server-bin /path/to/llama-server --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf
   ```
   It listens on `ws://0.0.0.0:8765`. Useful flags:
   - `--llm-threads 4`
   - `--tts-voice alba`
   - `--llama-url http://host:port` to use a llama-server you already run
   - `--gemini-key` for Gemini help
4. Keep the brain computer from going to sleep.

### Voice backend (Settings)
- **Auto** (default) uses the local brain first. If the brain can't be reached at start, or fails 3 times in a
  row, the app switches to Gemini Live, which needs an API key. It switches back when the brain is reachable and
  the robot is idle.
- **Local only** or **Gemini only** pin one backend.
- **Gemini for hard questions** applies when an API key is set. The local model can then pass maths, code and
  long explanations to Gemini, and the local voice still speaks the answer. Say "ask Gemini ..." to force it.

### Behaviour and speed
- **No made-up answers.** Requests for the camera, weather, web, FRC data, emotions, dances and music must call
  the matching tool first. The small model can't answer them from memory.
- **Spoken numbers.** Numbers are spoken as words. Team numbers are read digit by digit, so 254 is "two five four".
  Records like 9-1-0 are read as "nine wins, one loss, zero ties".
- **Speed.** These timings come from a CPU-only laptop (Intel i5-10310U), measured from when you stop talking
  to Reachy's first word:

  | Request | Time to first word |
  |---|---|
  | Music controls (pause, play, skip, stop) | about 1.5 s |
  | Chat | 3-5 s |
  | Play music, emotions | 7-14 s |
  | Weather, team lookup | about 16 s |
  | Web search | about 25 s |
  | Camera | about 60 s |
  | First start (one time) | about 80 s warm-up |

---

## FRC robotics helper (Nexus + The Blue Alliance)

This is for FIRST Robotics Competition events. It uses no Google services.

1. In **Settings, FRC robotics**, paste a free [Blue Alliance](https://www.thebluealliance.com/account) read key.
   It's used for team lookups, results and rankings. Optionally, also paste an [FRC Nexus](https://frc.nexus/api)
   API key for live queuing.
2. **Look up any team, no event needed.** Ask "tell me about team 1678". Reachy says the team's name, hometown and
   rookie year, its recent competitions and results, and its awards.
3. **At an event**, press **Scan QR code** (or say "scan this QR code"). Then hold up your team's Nexus code. It
   looks like `https://frc.nexus/<event>/<team>?key=...`. The code is decoded on the robot, and the event and team
   are filled in. Show one code at a time: if several are in view, Reachy asks for just yours. The private team
   link stays on the robot and is never spoken, shown or sent to any model.
4. Ask "When's our next match?", "What's queuing now?", "Where is team 254's pit?" or "Show me the rankings."
   With **Queuing alerts** on, Reachy says out loud when your match is queuing soon, now queuing or on deck. It
   checks Nexus every 30 seconds.

Match data comes from [FRC Nexus](https://frc.nexus) and [The Blue Alliance](https://www.thebluealliance.com).
