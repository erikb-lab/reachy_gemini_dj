"""IMU helpers, event watcher (synthetic sensor streams, fake clock), reactions, the tool's wording and brain intents."""
import math
import random

import pytest

from reachy_gemini_dj import imu as imu_mod
from reachy_gemini_dj.imu import (G, ImuReactions, ImuWatcher, accel_deviation_g, describe, motion_magnitude,
                                  posture_for_tilt, tilt_deg)

DT = 0.05            # 20 Hz


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def sample(accel=(-0.07, 0.85, 9.65), gyro=(0.0005, 0.001, 0.002), temp=45.7):
    return {"accelerometer": list(accel), "gyroscope": list(gyro), "quaternion": [1, 0, 0, 0], "temperature": temp}


def tilted(deg, mag=9.68):
    r = math.radians(deg)
    return (mag * math.sin(r), 0.0, mag * math.cos(r))


class Rig:
    """A watcher on a fake clock plus a stream generator."""

    def __init__(self, suppress=None, no_bump=None):
        self.clock = Clock()
        self.events = []
        self.rng = random.Random(7)
        self.w = ImuWatcher(read_imu=lambda: None, clock=self.clock, on_event=lambda k, i: self.events.append((k, i)),
                            suppress=suppress, no_bump=no_bump)

    def kinds(self):
        return [k for k, _ in self.events]

    def noise(self, s):
        return self.rng.gauss(0, s)

    def run(self, seconds, fn=None):
        """fn(t_rel) -> sample dict; default: at rest with sensor noise."""
        n = int(round(seconds / DT))
        for i in range(n):
            self.clock.t += DT
            t = i * DT
            self.w.feed(fn(t) if fn else self.rest_sample())

    def rest_sample(self, deg=5.0, temp=45.7):
        x, y, z = tilted(deg)
        return sample((x + self.noise(0.015), y + self.noise(0.015), z + self.noise(0.015)),
                      (self.noise(0.002), self.noise(0.002), self.noise(0.002)), temp)


# ------------------------------------------------------------------------------------------- helpers ---
def test_tilt_from_gravity():
    assert tilt_deg([0, 0, 9.81]) == pytest.approx(0, abs=0.01)
    assert tilt_deg([-0.07, 0.85, 9.65]) == pytest.approx(5.0, abs=0.2)           # the real robot at rest
    assert tilt_deg([9.81, 0, 0]) == pytest.approx(90, abs=0.01)
    assert tilt_deg([0, 0, -9.81]) == pytest.approx(180, abs=0.01)
    assert tilt_deg(None) is None and tilt_deg([1, 2]) is None and tilt_deg([0, 0, 0.1]) is None
    assert tilt_deg([float("nan"), 0, 9.8]) is None


def test_posture_words_and_hysteresis():
    assert posture_for_tilt(5) == "upright"
    assert posture_for_tilt(40) == "leaning"
    assert posture_for_tilt(90) == "on its side"
    assert posture_for_tilt(160) == "upside down"
    assert posture_for_tilt(26, "upright") == "upright"            # inside the band: no flicker
    assert posture_for_tilt(29, "upright") == "leaning"
    assert posture_for_tilt(25, "leaning") == "leaning"
    assert posture_for_tilt(21, "leaning") == "upright"
    assert posture_for_tilt(62, "leaning") == "leaning" and posture_for_tilt(66, "leaning") == "on its side"
    assert posture_for_tilt(60, "on its side") == "on its side" and posture_for_tilt(54, "on its side") == "leaning"


def test_motion_and_deviation():
    assert motion_magnitude([0.0005, 0.001, 0.002]) < 0.01
    assert motion_magnitude([3, 4, 0]) == 5
    assert motion_magnitude(None) == 0.0
    assert accel_deviation_g([0, 0, G]) == pytest.approx(0)
    assert accel_deviation_g([0, 0, 2 * G]) == pytest.approx(1.0)
    assert accel_deviation_g("junk") == 0.0


# --------------------------------------------------------------------------------- the event watcher ---
def test_at_rest_with_noise_is_silent_and_reports_state():
    r = Rig()
    r.run(30)
    assert r.events == []
    snap = r.w.snapshot()
    assert snap["posture"] == "upright" and snap["moving"] is False and snap["held"] is False
    assert snap["tilt_deg"] == pytest.approx(5.0, abs=1.0)
    assert snap["temperature_c"] == 45.7
    assert r.w.status()["available"] is True and r.w.status()["hot"] is False


def test_pickup_and_set_down():
    r = Rig()
    r.run(10)

    def carried(t):                    # lifted and tilted, hand jitter, ~3 s
        x, y, z = tilted(20 + 6 * math.sin(t * 2), mag=9.68 + 1.4 * math.sin(t * 5))
        return sample((x + r.noise(0.2), y + 1.0 * math.sin(t * 3), z), (0.6 * math.sin(t * 4), 0.4, 0.2 * math.cos(t * 3)))
    r.run(3, carried)
    assert r.kinds() == ["picked_up"]
    assert r.w.snapshot()["held"] is True
    assert r.events[0][1]["tilt_change"] >= 6 or r.events[0][1]["peak_dyn"] >= 0.8
    r.run(4)                           # put down, still again
    assert r.kinds() == ["picked_up", "set_down"]
    assert r.w.snapshot()["held"] is False


def test_short_ride_gets_no_set_down_and_slow_drift_is_not_a_pickup():
    r = Rig()
    r.run(10)
    r.run(8, lambda t: r.rest_sample(deg=5 + t * 1.5))          # table slowly tilted 12 degrees over 8 s: no motion
    assert r.events == []


def test_single_tap_is_a_bump_not_a_pickup():
    r = Rig()
    r.run(10)

    def tap(t):
        s = r.rest_sample()
        if 0.0 <= t < DT * 1.5:
            s["accelerometer"][2] += 3.0
            s["gyroscope"] = [0.5, 0.3, 0.1]
        return s
    r.run(3, tap)
    assert r.kinds() == ["bumped"]
    assert r.events[0][1]["peak_acc"] >= 0.8
    assert r.w.snapshot()["held"] is False


def test_sustained_shove_is_not_a_tap():
    r = Rig()
    r.run(10)
    r.run(2, lambda t: sample((0.0, 0.0, 9.68 + (1.5 if t < 0.5 else 0.0)), (0.02, 0.0, 0.0)) if t < 1 else r.rest_sample())
    assert "bumped" not in r.kinds()


def test_shaking():
    r = Rig()
    r.run(10)

    def shake(t):
        s = math.sin(2 * math.pi * 3 * t)
        return sample((6 * s, 3 * s, 9.7 + 5 * s), (3.0 * s, 2.0 * s, 1.0 * s))
    r.run(2.5, shake)
    assert r.kinds() == ["shaken"]            # no "picked up" chatter before it
    info = r.events[0][1]
    assert info["span_s"] >= 1.0 and info["reversals"] >= 4
    r.run(5)                                   # then put down: silent (no ride was announced)
    assert r.kinds() == ["shaken"]


def test_tipped_over_announces_after_two_seconds_at_most_twice():
    r = Rig()
    r.run(10)

    def fall(t):                               # a short tumble, then lying on its side
        if t < 0.4:
            return sample((3.0 * math.sin(t * 20), 4.0, 8.0), (1.5, 0.5, 0.2))
        x, y, z = tilted(88)
        return sample((x + r.noise(0.015), y + r.noise(0.015), z + r.noise(0.015)), (r.noise(0.002),) * 3)
    r.run(1.9, fall)
    assert "tipped" not in r.kinds()
    r.run(1.5, lambda t: fall(1.0))
    assert r.kinds().count("tipped") == 1
    r.run(25, lambda t: fall(1.0))            # still lying there: cooldown / repeat spacing
    assert r.kinds().count("tipped") == 1
    r.run(15, lambda t: fall(1.0))
    assert r.kinds().count("tipped") == 2
    r.run(80, lambda t: fall(1.0))
    assert r.kinds().count("tipped") == 2     # never a third
    r.run(5)                                   # righted
    r.run(15, lambda t: fall(1.0))
    assert r.kinds().count("tipped") == 3     # re-armed once it had been upright


def test_leaning_is_not_tipped():
    r = Rig()
    r.run(10)
    r.run(10, lambda t: r.rest_sample(deg=45))
    assert "tipped" not in r.kinds()
    assert r.w.snapshot()["posture"] == "leaning"


def test_upside_down_counts_as_tipped():
    r = Rig()
    r.run(10)
    r.run(4, lambda t: sample((0.0, 0.0, -9.68), (0.001, 0.0, 0.0)))
    assert "tipped" in r.kinds()
    assert r.w.snapshot()["posture"] == "upside down"


def test_own_motion_is_ignored_with_grace():
    moving = {"on": False}
    r = Rig(suppress=lambda: moving["on"])
    r.run(10)

    def violent(t):
        s = math.sin(2 * math.pi * 3 * t)
        return sample((6 * s, 3 * s, 9.7 + 5 * s), (3.0 * s, 2.0 * s, 1.0 * s))
    moving["on"] = True
    r.run(3, violent)                          # the robot dancing
    moving["on"] = False
    r.run(1.0, violent)                        # still inside the 1.5 s grace
    assert r.events == []
    r.run(4)
    assert r.events == []


def test_own_motion_start_aborts_a_pending_tap():
    moving = {"on": False}
    r = Rig(suppress=lambda: moving["on"])
    r.run(10)
    r.clock.t += DT
    s = r.rest_sample()
    s["accelerometer"][2] += 3.0
    r.w.feed(s)
    moving["on"] = True
    r.run(3)
    moving["on"] = False
    r.run(5)
    assert r.events == []


def test_quiet_mode_style_suppression_and_resume():
    quiet = {"on": True}
    r = Rig(suppress=lambda: quiet["on"])
    r.run(10)
    r.run(2, lambda t: sample((6 * math.sin(40 * t), 3, 9.7 + 5 * math.sin(40 * t)), (3 * math.sin(40 * t), 0, 0)))
    assert r.events == []
    quiet["on"] = False
    r.run(5)
    r.run(2, lambda t: sample((6 * math.sin(2 * math.pi * 3 * t), 3, 9.7 + 5 * math.sin(2 * math.pi * 3 * t)),
                              (3 * math.sin(2 * math.pi * 3 * t), 0, 0)))
    assert "shaken" in r.kinds()               # and it works again afterwards


def test_no_bump_while_the_robot_is_loud():
    loud = {"on": True}
    r = Rig(no_bump=lambda: loud["on"])
    r.run(10)

    def tap(t):
        s = r.rest_sample()
        if t < DT * 1.5:
            s["accelerometer"][2] += 3.0
        return s
    r.run(3, tap)
    assert r.events == []
    loud["on"] = False
    r.run(3, tap)
    assert r.kinds() == ["bumped"]


def test_cooldown_per_event_kind():
    r = Rig()
    r.run(10)

    def tap(t):
        s = r.rest_sample()
        if t < DT * 1.5:
            s["accelerometer"][2] += 3.0
        return s
    r.run(3, tap)
    r.run(3, tap)                              # a second tap 3 s later: inside the 20 s cooldown
    assert r.kinds() == ["bumped"]
    r.run(20, None)
    r.run(3, tap)
    assert r.kinds() == ["bumped", "bumped"]


def test_nothing_during_warmup():
    r = Rig()
    r.run(1)
    r.run(1, lambda t: sample((6, 3, 14), (3, 2, 1)))
    assert r.events == []


def test_none_and_malformed_samples():
    r = Rig()
    assert r.w.snapshot() is None and r.w.status() == {"available": False}
    for bad in (None, {}, {"accelerometer": None, "gyroscope": None}, {"accelerometer": [1, 2], "gyroscope": [0, 0, 0]},
                {"accelerometer": [0, 0, 0], "gyroscope": [0, 0, 0]}, "x", {"accelerometer": ["a", 0, 0], "gyroscope": [0, 0, 0]}):
        r.clock.t += DT
        r.w.feed(bad)
    assert r.w.snapshot() is None
    r.run(5)
    assert r.w.snapshot() is not None
    r.clock.t += imu_mod.STALE_S + 1                 # the sensor stops answering
    assert r.w.snapshot() is None and r.w.sensors()["error"]


def test_missing_temperature_and_gap_restart():
    r = Rig()
    s = sample()
    s["temperature"] = None
    r.clock.t += DT
    r.w.feed(s)
    assert r.w.snapshot()["temperature_c"] is None
    r.clock.t += 5                                    # a long stall must not be read as a huge dt
    r.w.feed(s)
    assert r.events == []


def test_thread_polls_reader_and_tolerates_errors():
    import time
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return sample()
    w = ImuWatcher(read_imu=reader, hz=200)
    w.start()
    try:
        deadline = time.time() + 3
        while time.time() < deadline and w.snapshot() is None:
            time.sleep(0.02)
        assert w.snapshot() is not None
    finally:
        w.stop()


# ----------------------------------------------------------------------------------------- the tool ---
def snap(**kw):
    d = {"tilt_deg": 5.0, "posture": "upright", "moving": False, "held": False, "temperature_c": 45.7}
    d.update(kw)
    return d


def test_tool_say_strings():
    assert describe(snap())["say"] == ("I'm sitting upright, tilted about 5 degrees, and I'm not moving. "
                                       "My internal temperature is 46 degrees Celsius.")
    assert describe(snap(tilt_deg=0.8))["say"].startswith("I'm sitting upright and level, and I'm not moving.")
    assert describe(snap(moving=True))["say"].split(",")[-1].startswith(" and I'm moving.")
    assert "being held" in describe(snap(held=True, moving=True))["say"]
    assert describe(snap(tilt_deg=41, posture="leaning"))["say"].startswith("I'm leaning about 40 degrees")
    assert describe(snap(tilt_deg=90, posture="on its side"))["say"].startswith("I'm on my side")
    assert describe(snap(tilt_deg=170, posture="upside down"))["say"].startswith("I'm upside down")
    hot = describe(snap(temperature_c=72.4))
    assert hot["say"].endswith("My internal temperature is 72 degrees Celsius, which is hot.") and hot["hot"] is True
    assert describe(snap(temperature_c=69.9))["hot"] is False
    cold = describe(snap(temperature_c=None))
    assert "temperature" not in cold["say"] and "temperature_c" not in cold
    d = describe(snap())
    assert d["tilt_deg"] == 5 and d["posture"] == "upright" and d["moving"] is False and d["temperature_c"] == 45.7


def test_tool_errors_without_imu():
    assert "error" in describe(None)
    assert "can't read" in describe(None)["error"]
    w = ImuWatcher(read_imu=lambda: None)
    assert "error" in w.sensors()


def test_robot_sensors_declared_and_info():
    from reachy_gemini_dj.tool_schemas import INFO_TOOLS, robot_tool_specs
    assert "robot_sensors" in INFO_TOOLS
    spec = next((s for s in robot_tool_specs() if s["name"] == "robot_sensors"), None)
    assert spec and spec["kind"] == "info" and spec["parameters"].get("properties") == {}
    assert len(spec["description"]) < 130


def test_gemini_execute_tool_dispatch():
    from reachy_gemini_dj.gemini import GeminiSession
    sess = GeminiSession.__new__(GeminiSession)
    sess.on_event = lambda k, t: None
    assert "error" in sess.execute_tool("robot_sensors", {})
    clock = Clock()
    w = ImuWatcher(read_imu=lambda: None, clock=clock)
    clock.t += 1
    w.feed(sample())
    sess.imu = w
    res = sess.execute_tool("robot_sensors", {})
    assert res["say"].startswith("I'm sitting upright")


# ------------------------------------------------------------------------------------------ reactions ---
def make_reactions(**kw):
    said, moves = [], []
    clock = Clock()
    args = dict(say=said.append, emotion=moves.append, clock=clock, rng=random.Random(1))
    args.update(kw)
    return ImuReactions(**args), said, moves, clock


def test_reaction_lines():
    rx, said, moves, clock = make_reactions()
    for kind in ("picked_up", "set_down", "bumped", "shaken", "tipped"):
        assert rx.on_event(kind, {})
        clock.t += 10
    assert said == ["Whoa, where are we going?", "Thanks for the ride.", "Hey!", "I'm getting dizzy!",
                    "I fell over. Can you help me up?"]
    assert len(moves) == 1 and moves[0] in ("surprised", "attentive")      # only the pickup moves


def test_reactions_never_stack_or_interrupt():
    rx, said, _, clock = make_reactions()
    assert rx.on_event("picked_up") is True
    clock.t += 1
    assert rx.on_event("shaken") is False                                    # too soon after the last one
    clock.t += 10
    assert rx.on_event("shaken") is True
    assert len(said) == 2
    assert rx.on_event("nonsense") is False


def test_reactions_respect_speaking_quiet_toggle_and_motion():
    busy = {"speaking": True, "quiet": True, "on": False, "free": False}
    rx, said, moves, clock = make_reactions(is_idle=lambda: not busy["speaking"], is_quiet=lambda: busy["quiet"],
                                            enabled=lambda: busy["on"], motion_free=lambda: busy["free"])
    assert rx.on_event("bumped") is False
    busy["on"] = True
    assert rx.on_event("bumped") is False                                    # quiet
    busy["quiet"] = False
    assert rx.on_event("bumped") is False                                    # speaking
    busy["speaking"] = False
    assert rx.on_event("picked_up") is True and said == ["Whoa, where are we going?"]
    assert moves == []                                                       # motion not free: line only
    clock.t += 10
    busy["free"] = True
    rx.on_event("picked_up")
    assert len(moves) == 1


def test_reactions_survive_failures():
    def boom(_):
        raise RuntimeError("no voice")
    rx, _, _, _ = make_reactions(say=boom)
    assert rx.on_event("bumped") is False
    rx2, said, _, _ = make_reactions(emotion=boom)
    assert rx2.on_event("picked_up") is True and said


def test_watcher_to_reactions_end_to_end():
    said = []
    clock = Clock()
    rx = ImuReactions(say=said.append, clock=clock)
    r = Rig()
    r.w.on_event = rx.on_event
    rx.clock = r.clock
    r.run(10)

    def tap(t):
        s = r.rest_sample()
        if t < DT * 1.5:
            s["accelerometer"][2] += 3.0
        return s
    r.run(3, tap)
    assert said == ["Hey!"]


def test_config_default_and_motion_moving_flag():
    from reachy_gemini_dj.config import Config
    assert Config().imu_reactions is True
    import threading
    from reachy_gemini_dj.motion import Motion

    class Mini:
        pass
    m = Motion(Mini(), threading.Event())
    assert m.moving() is False
    m.learning = True
    assert m.moving() is True
    m.learning = False
    m.busy = True
    assert m.moving() is True
    m._stop.set()


# ------------------------------------------------------------------------------------------ brain intents ---
ALLT = {"robot_sensors", "get_weather", "web_search", "look", "robot_emotion", "play_music", "set_timer"}


@pytest.mark.parametrize("text", [
    "How hot are you?", "What's your temperature?", "what is your internal temperature", "Are you level?",
    "are you tilted", "Are you upright?", "what does your IMU say", "read your sensors", "check your IMU",
    "Are you being picked up?", "which way is up", "Are you hot?", "what's your current temperature",
    "what does your sensor say", "read your I M U", "how's your gyroscope"])
def test_sensor_intents_are_forced(text):
    from reachy_brain.session import forced_tool
    assert forced_tool(text, ALLT) == "robot_sensors"


@pytest.mark.parametrize("text", [
    "What's the temperature outside?", "What's the temperature in Chicago?", "what's the temperature today",
    "How hot is it outside", "what is the temperature tomorrow in Detroit", "Is it going to rain? What's the forecast",
    "What's the weather like?", "what's your temperature outside today", "how cold is it"])
def test_outdoor_temperature_still_goes_to_weather(text):
    from reachy_brain.session import forced_tool
    assert forced_tool(text, ALLT) == "get_weather"


def test_sensor_intent_negatives():
    from reachy_brain.session import forced_tool
    assert forced_tool("what do you see in front of you", ALLT) == "look"
    assert forced_tool("tell me a joke", ALLT) is None
    assert forced_tool("how are you", ALLT) is None
    assert forced_tool("are you happy", ALLT) is None
    assert forced_tool("Show me that you're really happy.", ALLT) == "robot_emotion"
    assert forced_tool("who won the world series this year", ALLT) == "web_search"
    assert forced_tool("how hot are you", ALLT - {"robot_sensors"}) is None      # tool not offered: let the model say so
    assert forced_tool("play some daft punk", ALLT) == "play_music"


def test_sensor_quick_commands():
    from reachy_brain.session import quick_command
    for t in ("how hot are you", "How hot are you?", "what's your temperature", "What is your temperature right now?",
              "read your IMU", "Can you read your sensors please", "what does your IMU say", "check your sensors"):
        assert quick_command(t) == ("robot_sensors", "Let me check.", {}), t
    for t in ("what's the temperature outside", "what's the temperature in Chicago", "how hot is it", "are you level",
              "what's your temperature outside", "how hot are you going to be tomorrow", "check your email"):
        q = quick_command(t)
        assert q is None or q[0] != "robot_sensors", t
