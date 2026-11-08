---
title: Reachy Gemini DJ
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
short_description: Voice DJ, 360 photos and helper for Reachy Mini, free local voice
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
  volume while you talk (in push-to-talk mode, only while talk is held or Reachy speaks). A 3-band **Music EQ**
  with presets is in Settings. "Pause", "play the music", "skip" and "stop the music" work instantly.
- **Moves and emotions.** Emotions, dances and head moves come from Pollen's official libraries. **Learn poses**
  lets you turn the motors off, pose Reachy by hand, lock the pose and save it by name. Then say "go to my
  playing-music pose".
- **Gamepad mode (experimental).** Drive the motors directly with a game controller (Xbox, PlayStation, Switch Pro,
  8BitDo…). Turn it on in **Settings > Experimental** (speed and button mapping live there too), pair the controller
  over Bluetooth with the phone or computer showing the controller page, open the **Robot** tab, press any button,
  then **Start gamepad mode** (or Options/Start on the controller). Sticks move the head and body, bumpers tilt,
  triggers lean, the D-pad moves the antennas, A saves the pose, B centres, Y toggles fine control. Moves are
  speed-capped and limited to a safe range; other moves pause while it is on, and it turns itself off after
  3 minutes without input.
  Limits follow Pollen's documented ranges (head pitch/roll ±40°, head yaw ±180°, body ±160°, head within 65° of the
  body). The **Motor health** card on the Robot tab shows motor errors and clears an overload without a power cycle.
- **Camera.** Ask "what do you see?" and the local vision model describes what is in front of Reachy.
- **360° photos.** Say "take a 360 photo" (or press **Take 360° photo** on the Robot tab). Reachy turns all the way
  around, takes sharp full-resolution photos, and stitches one 360° picture on the robot, up to 6144 px wide. You
  choose how many photos to take, from a 2-minute preview to an 80-photo slow mode
  ([details below](#360-photos)).
- **Weather and web search.** Both are free (Open-Meteo and DuckDuckGo) and need no keys.
- **FRC helper.** Scan your team's FRC Nexus QR code, hear queuing alerts, and look up any team's history and
  results ([below](#frc-robotics-helper-nexus--the-blue-alliance)).
- **Timers, reminders and alarms.** Say "set a timer for 10 minutes", "remind me in 20 minutes to check the
  battery" or "wake me at 7:30". They are kept on the robot, survive a restart, and ring with a chime, a little
  move and a spoken message until you speak, press talk, or say "stop the alarm".
- **Memory.** Say "remember that our pit is A3", "what do you remember?" or "forget the pit". Facts are stored on
  the robot, and you can see, add and delete them in the **Memory** card in Settings.
- **Push-to-talk for loud rooms.** In Settings, set **Listening** to push-to-talk. Then hold the big button on the
  controller page (or the Space bar) while you talk. A future physical button can use the same `/api/ptt` endpoint
  ([below](#push-to-talk-api)).
- **Google Calendar, Tasks and Gmail (optional, read-only for Gmail).** Connect them in the **Google** tab.
- **Controller web page** at `http://reachy-mini.local:8042`. It has a dark mode, live volume sliders for the
  music and for Reachy's voice, and works on phones.

---

## 360° photos

Reachy turns on the spot, looks up, level and down, and stitches the photos into one 360° picture (an equirectangular
JPEG with Google "GPano" metadata, so photo viewers treat it as a photo sphere). Pictures show up in the **Robot** tab:
a gallery, a full-screen viewer you can drag around, **Download** and **Delete**.

**Start one:** press **Take 360° photo** on the Robot tab, or say "take a 360 photo". Pick a mode first, or say it:
"take a fast 360 photo", "take the best-quality one".

| Mode | Photos | About how long (estimate) | Use it for |
|---|---|---|---|
| Ultra fast | 15 | 2 min | a quick preview; seams and steps are likely |
| Fast | 25 | 3 min | most rooms |
| Regular (default) | 42 | 5 min | everyday use |
| Slow | 60 | 7 min | smaller seam steps on near objects |
| Ultra slow | 80 | 7 min | the cleanest result; the picture is 4096 px wide to fit in memory |

More photos give a cleaner result because neighbouring photos are taken closer together. Times are for the robot's
stitching; the capture part is about 2.5 s a photo.

**Good to know**
- **Put Reachy on a flat surface** and step out of the picture. The robot turns its whole body, so keep the area clear.
- **Live video pauses** while the photos are taken: the app asks the robot for exclusive use of the camera, then gives
  it back.
- Photos are taken at full resolution with the camera's whole field of view, with autofocus. Reachy waits for each
  picture to be sharp before it takes it.
- Everything stays on the robot, in `~/.reachy_gemini_dj/panoramas`. The newest 30 pictures are kept. **Delete** removes
  a picture, its thumbnail and its raw photos.
- The picture size adapts to the robot's free memory (8192, 6144 or 4096 px wide).
- Movements stay inside Pollen's documented limits, a little under the edge so the head does not press against it.

**Known limits.** Near objects (a floor, a mattress edge) can show small steps in the seams, because the camera sits a
few centimetres from the head's turning point. More photos make the steps smaller, not zero. The very top and bottom
of the sphere are filled in smoothly, because the head cannot look straight up or down.

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

### Run the brain as a service
On a Linux laptop with systemd you can let the brain start by itself and restart if it crashes. Run
`tools/install_brain_service.sh` with the paths to the python venv, `llama-server`, the model and the mmproj
(`--python`, `--llama-server`, `--model`, `--mmproj`). It writes a user service called `reachy-brain` and starts it.
Add `--at-boot` to start it at boot instead of at login. Logs go to the journal:
`journalctl --user -u reachy-brain -f`. The first start takes a couple of minutes while the models load.
The undo steps are at the bottom of the script.

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
  Records like 9-1-0 are read as "nine wins, one loss, zero ties", and pit codes like A3 as "A three".
- **Ready-made answers.** Weather, team lookups, the next-match briefing, rankings, timers and memory build their
  spoken answer in code, so they skip the slow second pass through the language model. For slow lookups Reachy
  says "Let me check the weather" (or similar) straight away and tilts its head while it thinks.
- **Fast restarts.** The brain saves its warmed-up prompt to disk (`--slot-cache-dir`, default
  `~/.cache/reachy_brain/slots`, about 200 MB per file, newest 3 kept), so a restart skips the long warm-up.
- **Speed.** These timings come from a CPU-only laptop (Intel i5-10310U), measured from when you stop talking
  to Reachy's first word:

  | Request | First word | Full answer starts |
  |---|---|---|
  | Music controls, set or cancel a timer | about 1.5 s | same |
  | Chat | 3-4 s | same |
  | Memory ("remember…", "what do you remember?") | 7-12 s | same |
  | Play music, emotions | 7-14 s | same |
  | Weather, team lookup, next match | about 1 s ("Let me check…") | 10-12 s |
  | Web search | about 1 s ("Let me look that up") | about 23 s |
  | Camera | about 1 s ("Let me take a look") | about 50 s |
  | Brain restart | under 1 s (saved warm-up) | |
  | Very first start, or after changing tools or settings that reach the brain | about 110 s warm-up | |

### Privacy and network safety
- **The Gemini API key never leaves the robot.** For "ask Gemini", the brain asks the robot, and the robot calls
  Gemini itself. Only the answer goes back to the brain.
- **The robot only runs tools it offered.** It ignores any other tool a brain asks for.
- **Private data and the camera need someone to have spoken.** Gmail, Calendar, Tasks, QR scanning and camera
  images are only given out within 2 minutes of someone speaking to the robot. A brain can't read them silently in
  the background. Every camera image sent to the brain shows up in the event feed, and so does every blocked request.
- **Memory stays on the robot.** Remembered facts are stored on the robot and sent only to the brain (or Gemini)
  you configured, so it can use them. The app logs how many facts there are, never what they say.
- **The brain gets your memory and notes.** Whoever controls the **Brain URL** setting receives remembered facts
  and your extra instructions, so every change to it is shown in the event feed.
- **Use a network you trust.** The Reachy Mini's own control pages, like this app's settings page and the
  push-to-talk endpoint, have no password. Anyone on the same Wi-Fi can change the settings. Keep the robot and the
  brain off guest or public networks.

### Push-to-talk API
For a future hardware button (an ESP32 button, a foot pedal bridge, and so on). Set **Listening** to push-to-talk
first.
- `POST http://reachy-mini.local:8042/api/ptt` with `{"down": true}` when pressed and `{"down": false}` when released.
- `POST http://reachy-mini.local:8042/api/ptt/toggle` for one-button devices: tap to start, tap to stop.
- Pressing talk also stops Reachy mid-sentence and silences a ringing alarm. A press held for 30 seconds releases
  itself.

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
4. Ask "Tell me about our next match" (partners and opponents with their names, records and ranks), "What's
   queuing now?", "Where is team 254's pit?" or "Show me the rankings." Blue Alliance answers are cached for a
   minute at events and longer for team history, so repeat questions are instant and the free API isn't overused.
   With **Queuing alerts** on, Reachy says out loud when your match is queuing soon, now queuing or on deck. It
   checks Nexus every 30 seconds.

Match data comes from [FRC Nexus](https://frc.nexus) and [The Blue Alliance](https://www.thebluealliance.com).
