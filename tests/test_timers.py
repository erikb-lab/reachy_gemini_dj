"""Timers, reminders and alarms: parsing, tools, persistence, ringing (fake clock, no audio, no threads)."""
import datetime
import json

import pytest

from reachy_gemini_dj import timers as T
from reachy_gemini_dj.timers import Timers, parse_clock, parse_duration


def epoch(h=15, m=0, s=0, day=10):
    return datetime.datetime(2026, 3, day, h, m, s).timestamp()


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def adv(self, secs):
        self.t += secs


class Spy:
    def __init__(self):
        self.said, self.events, self.chimes, self.emotions = [], [], [], []
        self.ducks, self.resets = [], 0

    # say / on_event / play_pcm / motion / music in one object
    def say(self, t): self.said.append(t)
    def on_event(self, k, t): self.events.append((k, t))
    def play_pcm(self, b): self.chimes.append(len(b))
    def emotion(self, n): self.emotions.append(n)
    def set_duck(self, v): self.ducks.append(v)
    def reset_duck(self): self.resets += 1


def make(tmp_path, now=None, **kw):
    spy = Spy()
    clock = Clock(now or epoch())
    t = Timers(say=spy.say, on_event=spy.on_event, motion=spy, music=spy, clock=clock, path=tmp_path / "timers.json",
               play_pcm=spy.play_pcm, **kw)
    return t, clock, spy


# ------------------------------------------------------------------------------- durations ---
@pytest.mark.parametrize("text,secs", [
    ("10 minutes", 600), ("1 hour 30 minutes", 5400), ("an hour and a half", 5400), ("half an hour", 1800),
    ("90 seconds", 90), ("5 min", 300), ("a minute", 60), ("2 hours", 7200), ("one and a half minutes", 90),
    ("ten minutes", 600), ("twenty five minutes", 1500), ("forty-five seconds", 45), ("sixty seconds", 60),
    ("1 hour and 30 minutes", 5400), ("1.5 hours", 5400), ("a quarter of an hour", 900), ("three quarters of an hour", 2700),
    ("an hour", 3600), ("half a minute", 30), ("a couple of minutes", 120), ("1h30m", 5400), ("90s", 90),
    ("for 10 minutes", 600), ("in 20 minutes", 1200), ("a timer for 10 minutes", 600), ("15", 900), ("ten", 600),
    ("two hours and fifteen minutes", 8100), ("an hour and a half", 5400), ("one hour thirty minutes", 5400),
    ("10 MINUTES", 600), ("5 mins", 300), ("2 hrs", 7200), ("one hundred twenty seconds", 120),
    ("a half hour", 1800), ("30 secs", 30),
])
def test_parse_duration(text, secs):
    assert parse_duration(text) == secs


@pytest.mark.parametrize("text", ["", "soon", "banana", "0 minutes", "7:30", "minutes", "ten bananas", "3 days", "half"])
def test_parse_duration_rejects(text):
    with pytest.raises(ValueError) as e:
        parse_duration(text)
    assert str(e.value).endswith((".", "?"))              # speakable sentence


# ------------------------------------------------------------------------------ clock times ---
def at(text, now=None):
    fire, h, m = parse_clock(text, now or epoch(15, 0))
    return datetime.datetime.fromtimestamp(fire), h, m


def test_clock_basic_phrasings():
    now = epoch(15, 0)                                    # 3:00 PM on the 10th
    assert at("7:30 pm", now)[0] == datetime.datetime(2026, 3, 10, 19, 30)
    assert at("19:30", now)[0] == datetime.datetime(2026, 3, 10, 19, 30)
    assert at("7 am", now)[0] == datetime.datetime(2026, 3, 11, 7, 0)
    assert at("7:30 a.m.", now)[0] == datetime.datetime(2026, 3, 11, 7, 30)
    assert at("noon", now)[0] == datetime.datetime(2026, 3, 11, 12, 0)             # today's noon has passed
    assert at("midnight", now)[0] == datetime.datetime(2026, 3, 11, 0, 0)
    assert at("6:45 in the morning", now)[0] == datetime.datetime(2026, 3, 11, 6, 45)
    assert at("7 tonight", now)[0] == datetime.datetime(2026, 3, 10, 19, 0)
    assert at("7:30", now)[0] == datetime.datetime(2026, 3, 10, 19, 30)            # next 7:30 is this evening
    assert at("7:30", epoch(21, 0))[0] == datetime.datetime(2026, 3, 11, 7, 30)    # at 9 PM it is tomorrow morning
    assert at("7:30", epoch(5, 0))[0] == datetime.datetime(2026, 3, 10, 7, 30)
    assert at("tomorrow at 7 am", now)[0] == datetime.datetime(2026, 3, 11, 7, 0)
    assert at("tomorrow 8:15 pm", epoch(7, 0))[0] == datetime.datetime(2026, 3, 11, 20, 15)
    assert at("5 o'clock in the afternoon", epoch(9, 0))[0] == datetime.datetime(2026, 3, 10, 17, 0)
    assert at("8 in the evening", now)[0] == datetime.datetime(2026, 3, 10, 20, 0)
    assert at("12 am", now)[0] == datetime.datetime(2026, 3, 11, 0, 0)
    assert at("12:30 pm", epoch(9, 0))[0] == datetime.datetime(2026, 3, 10, 12, 30)
    assert at("0730", epoch(1, 0))[0] == datetime.datetime(2026, 3, 10, 7, 30)
    assert at("1930", now)[0] == datetime.datetime(2026, 3, 10, 19, 30)
    assert at("11 at night", now)[0] == datetime.datetime(2026, 3, 10, 23, 0)
    assert at("3 pm", epoch(14, 59, 30))[0] == datetime.datetime(2026, 3, 10, 15, 0)
    assert at("3 pm", epoch(15, 0))[0] == datetime.datetime(2026, 3, 11, 15, 0)    # exactly now means next time


def test_clock_spelled_phrasings():
    now = epoch(15, 0)
    assert at("seven thirty", now)[0] == datetime.datetime(2026, 3, 10, 19, 30)
    assert at("seven oh five am", now)[0] == datetime.datetime(2026, 3, 11, 7, 5)
    assert at("half past six", now)[0] == datetime.datetime(2026, 3, 10, 18, 30)
    assert at("quarter to seven pm", now)[0] == datetime.datetime(2026, 3, 10, 18, 45)
    assert at("quarter past 7 am", now)[0] == datetime.datetime(2026, 3, 11, 7, 15)
    assert at("ten to eight am", now)[0] == datetime.datetime(2026, 3, 11, 7, 50)
    assert at("six forty five in the morning", now)[0] == datetime.datetime(2026, 3, 11, 6, 45)
    assert at("set an alarm to 6 am", now)[0] == datetime.datetime(2026, 3, 11, 6, 0)


@pytest.mark.parametrize("text", ["", "banana", "25:00", "7:75", "soon", "99", "7:30:15 elephants"])
def test_clock_rejects(text):
    with pytest.raises(ValueError):
        parse_clock(text, epoch())


# --------------------------------------------------------------------------------- tools ---
def test_set_timer_say(tmp_path):
    t, clock, _ = make(tmp_path)
    assert t.set_timer("10 minutes")["say"] == "Timer set for 10 minutes."
    assert t.set_timer("an hour and a half")["say"] == "Timer set for 1 hour and 30 minutes."
    assert t.set_timer("90 seconds")["say"] == "Timer set for 1 minute and 30 seconds."
    assert t.set_timer("4 minutes", "pizza")["say"] == "Timer set for 4 minutes, called pizza."


def test_reminder_say(tmp_path):
    t, clock, _ = make(tmp_path, epoch(15, 25))
    res = t.set_timer("20 minutes", "check the battery")
    assert res["say"] == "Okay, I'll remind you to check the battery at 3:45 PM."
    assert t.set_timer("5 minutes", "to call mom")["say"] == "Okay, I'll remind you to call mom at 3:30 PM."


def test_set_alarm_say(tmp_path):
    t, clock, _ = make(tmp_path, epoch(21, 0))
    assert t.set_alarm("7:30 am")["say"] == "Alarm set for 7:30 AM tomorrow."
    assert t.set_alarm("10 pm")["say"] == "Alarm set for 10 PM."
    assert t.set_alarm("noon")["say"] == "Alarm set for noon tomorrow."
    assert t.set_alarm("6 am", "run")["say"] == "Alarm set for 6 AM tomorrow, called run."
    assert t.set_alarm("7 am", "take out the trash")["say"] == "Okay, I'll remind you to take out the trash at 7 AM tomorrow."


def test_errors_are_speakable(tmp_path):
    t, _, _ = make(tmp_path)
    for res in (t.set_timer("banana"), t.set_timer(""), t.set_alarm("banana"), t.set_alarm("")):
        assert set(res) == {"error"} and res["error"].startswith("Sorry")
    assert "24 hours" in t.set_timer("30 hours")["error"]
    assert t.cancel_timer("")["error"] == "You have no timers or alarms to cancel."


def test_list(tmp_path):
    t, clock, _ = make(tmp_path, epoch(15, 0))
    assert t.list_timers() == {"timers": [], "say": "You have no timers or alarms."}
    t.set_timer("4 minutes", "pizza")
    assert t.list_timers()["say"] == "You have one timer: 4 minutes left on the pizza timer."
    t.set_alarm("7:30 am")
    assert t.list_timers()["say"] == ("You have two timers: 4 minutes left on the pizza timer, "
                                      "and an alarm at 7:30 AM tomorrow.")
    t.set_timer("1 hour 5 minutes")
    t.set_timer("20 minutes", "check the battery")
    assert t.list_timers()["say"] == (
        "You have four timers: 4 minutes left on the pizza timer, a reminder to check the battery in 20 minutes, "
        "1 hour and 5 minutes left on a timer, and an alarm at 7:30 AM tomorrow.")
    clock.adv(61)
    assert "3 minutes left on the pizza timer" in t.list_timers()["say"]
    assert [x["id"] for x in t.list_timers()["timers"]][0] == 1


def test_list_only_alarms_and_seconds(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_alarm("9 pm")
    assert t.list_timers()["say"] == "You have one alarm: an alarm at 9 PM."
    t.set_timer("45 seconds")
    assert "45 seconds left on a timer" in t.list_timers()["say"]


def test_cancel(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_timer("4 minutes", "pizza")
    t.set_timer("10 minutes")
    t.set_alarm("7 am")
    assert t.cancel_timer("pizza")["say"] == "Okay, cancelled the pizza timer."
    assert t.cancel_timer("the pizza timer")["error"].startswith("I couldn't find")
    assert t.cancel_timer("10 minutes")["say"] == "Okay, cancelled the 10 minute timer.".replace("minute ", "minutes ")
    assert t.cancel_timer("7 am")["say"].startswith("Okay, cancelled the 7 AM alarm")
    assert t.list_timers()["timers"] == []


def test_cancel_latest_and_all(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_timer("5 minutes")
    clock.adv(1)
    t.set_timer("6 minutes")
    assert t.cancel_timer("")["say"] == "Okay, cancelled the 6 minutes timer."
    assert len(t.list_timers()["timers"]) == 1
    t.set_timer("7 minutes")
    t.set_alarm("7 am")
    assert t.cancel_timer("all")["say"] == "Okay, cancelled all 3 timers and alarms.".replace("3", "three")
    assert t.list_timers()["timers"] == []
    t.set_alarm("7 am")
    t.set_alarm("8 am")
    assert t.cancel_timer("all alarms")["say"] == "Okay, cancelled both alarms."


def test_cancel_by_id(tmp_path):
    t, _, _ = make(tmp_path)
    tid = t.set_timer("5 minutes")["id"]
    assert t.cancel_id(tid)["ok"]
    assert "error" in t.cancel_id(tid)


def test_handle_dispatch(tmp_path):
    t, _, _ = make(tmp_path)
    assert t.handle("set_timer", {"duration": "2 minutes"})["say"] == "Timer set for 2 minutes."
    assert t.handle("set_alarm", {"time": "6 pm", "label": ""})["say"] == "Alarm set for 6 PM."
    assert "timer" in t.handle("list_timers", {})["say"]
    assert t.handle("cancel_timer", {})["say"].startswith("Okay")
    assert t.handle("stop_alarm", {})["say"] == "Nothing is ringing right now."
    assert "error" in t.handle("nope", {})


def test_limit(tmp_path):
    t, _, _ = make(tmp_path)
    for _ in range(T.MAX_ACTIVE):
        assert "error" not in t.set_timer("1 hour")
    assert "too many" in t.set_timer("1 hour")["error"]


# ------------------------------------------------------------------------------ persistence ---
def test_persistence_round_trip(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_timer("10 minutes", "pizza")
    t.set_alarm("7 am")
    t.set_timer("5 minutes", "call mom")
    clock.adv(60)
    t2, _, _ = make(tmp_path, now=clock.t)
    assert len(t2.status()["timers"]) == 3
    assert t2.list_timers()["say"] == ("You have three timers: a reminder to call mom in 4 minutes, "
                                       "9 minutes left on the pizza timer, and an alarm at 7 AM tomorrow.")
    t2.set_timer("1 minute")                                   # ids keep counting
    assert max(r["id"] for r in t2.status()["timers"]) == 4


def test_expired_while_off_dropped_recent_announced_as_missed(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_timer("2 minutes", "pizza")           # missed 3 minutes ago at restart
    t.set_timer("1 minute", "laundry")          # missed 20 minutes ago: dropped silently
    t.set_alarm("11 pm")
    clock.adv(0)
    old = json.loads((tmp_path / "timers.json").read_text())
    old["timers"][1]["fire_at"] = clock.t - 20 * 60
    old["timers"][0]["fire_at"] = clock.t - 3 * 60
    (tmp_path / "timers.json").write_text(json.dumps(old))
    t2, clock2, spy = make(tmp_path, now=clock.t, missed_delay_s=15)
    t2._missed_at = clock2.t + 15
    assert len(t2.status()["timers"]) == 1                       # only the alarm is still waiting
    t2.tick()
    assert spy.said == []                                         # not yet: the voice backend may still be connecting
    clock2.adv(16)
    t2.tick()
    assert spy.said == ["Sorry, I missed your pizza timer while I was off. It was due 3 minutes ago."]
    t2.tick()
    assert len(spy.said) == 1
    t3, _, spy3 = make(tmp_path, now=clock2.t)                   # announced once, then forgotten
    assert t3._missed == []


def test_corrupt_file_ignored(tmp_path):
    (tmp_path / "timers.json").write_text("{nope")
    t, _, _ = make(tmp_path)
    assert t.list_timers()["timers"] == []
    assert t.set_timer("1 minute")["ok"]


# ------------------------------------------------------------------------------------ ringing ---
def test_fire_repeat_and_expire(tmp_path):
    t, clock, spy = make(tmp_path)
    t.set_timer("10 minutes")
    clock.adv(599)
    t.tick()
    assert spy.said == [] and not t.status()["ringing"]
    clock.adv(1)
    t.tick()
    assert spy.said == ["Your 10 minutes timer is done."]
    assert spy.chimes and spy.emotions == ["surprised"] and spy.ducks == [T.DUCK_LEVEL]
    assert t.status()["ringing"] and t.status()["timers"][0]["ringing"]
    assert json.loads((tmp_path / "timers.json").read_text())["timers"] == []     # fired ones are not persisted
    clock.adv(10)
    t.tick()
    assert len(spy.said) == 1
    clock.adv(10)
    t.tick()
    assert len(spy.said) == 2
    clock.adv(20)
    t.tick()
    assert len(spy.said) == 3 and len(spy.chimes) == 3
    clock.adv(19)
    t.tick()
    assert t.status()["ringing"]
    clock.adv(1)
    t.tick()
    assert len(spy.said) == 3 and not t.status()["ringing"]       # gave up after three
    assert spy.resets >= 1                                         # music un-ducked


def test_ring_texts(tmp_path):
    t, clock, spy = make(tmp_path, epoch(7, 0))
    t.set_timer("1 minute", "check the battery")
    t.set_timer("1 minute", "pizza")
    t.set_alarm("7:30 am")
    clock.adv(60)
    t.tick()
    assert spy.said == ["Reminder: check the battery.", "Your pizza timer is done."]
    t.dismiss()
    spy.said.clear()
    clock.adv(30 * 60)
    t.tick()
    assert spy.said == ["It's 7:30 AM. This is your alarm."]


def test_stop_alarm_dismisses(tmp_path):
    t, clock, spy = make(tmp_path)
    t.set_timer("1 minute")
    t.set_timer("1 minute", "tea")
    clock.adv(60)
    t.tick()
    assert t.status()["ringing"]
    assert t.stop_alarm()["say"] == "Okay, stopped."
    assert not t.status()["ringing"] and spy.resets == 1
    clock.adv(10)
    t.tick()
    assert len(spy.said) == 2                                      # no repeats after dismissal
    clock.adv(15)
    t.tick()
    assert len(spy.said) == 2
    assert t.stop_alarm()["say"] == "Okay."                        # just dismissed: still polite
    clock.adv(60)
    assert t.stop_alarm()["say"] == "Nothing is ringing right now."


def test_cancel_with_nothing_waiting_stops_ringing(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_timer("1 minute")
    clock.adv(60)
    t.tick()
    assert t.cancel_timer("")["say"] == "Okay, stopped."
    assert not t.status()["ringing"]


def test_user_speech_dismisses(tmp_path):
    age = {"v": 1e9}
    t, clock, spy = make(tmp_path, user_speech_age=lambda: age["v"])
    t.set_timer("1 minute")
    clock.adv(60)
    t.tick()                                                       # rings; user last spoke long ago
    assert t.status()["ringing"]
    clock.adv(1)
    age["v"] = 0.5                                                 # speech right after the announcement: our echo
    t.tick()
    assert t.status()["ringing"]
    clock.adv(10)
    age["v"] = 4.0                                                 # user spoke 4 s ago, >2.5 s after the announcement
    t.tick()
    assert not t.status()["ringing"] and len(spy.said) == 1
    clock.adv(30)
    t.tick()
    assert len(spy.said) == 1


def test_speech_before_the_ring_does_not_dismiss(tmp_path):
    age = {"v": 0.0}
    t, clock, spy = make(tmp_path, user_speech_age=lambda: age["v"])
    age["v"] = 5.0
    t.set_timer("1 minute")
    clock.adv(60)
    age["v"] = 65.0                                                # last spoke before it fired
    t.tick()
    assert t.status()["ringing"]


def test_on_user_speech_method(tmp_path):
    t, clock, _ = make(tmp_path)
    t.set_timer("1 minute")
    clock.adv(60)
    t.tick()
    t.on_user_speech()
    assert not t.status()["ringing"]


def test_no_optional_helpers_needed(tmp_path):
    t = Timers(clock=Clock(epoch()), path=tmp_path / "x.json")
    t.set_timer("1 second")
    t.clock.adv(2)
    t.tick()                                                       # no say/motion/music/audio: must not raise
    assert t.status()["ringing"]


def test_helper_failures_do_not_break_ringing(tmp_path):
    class Boom:
        def emotion(self, n): raise RuntimeError("x")
        def set_duck(self, v): raise RuntimeError("x")
        def reset_duck(self): raise RuntimeError("x")

    def bad(_): raise RuntimeError("x")
    spy = Spy()
    clock = Clock(epoch())
    t = Timers(say=spy.say, motion=Boom(), music=Boom(), clock=clock, path=tmp_path / "x.json", play_pcm=bad)
    t.set_timer("1 second")
    clock.adv(2)
    t.tick()
    assert spy.said == ["Your 1 second timer is done."]
    t.dismiss()


def test_chime_is_valid_pcm():
    pcm = T._make_chime()
    assert len(pcm) % 2 == 0 and 0.3 < len(pcm) / 2 / T.CHIME_SR < 1.0
    assert max(abs(int.from_bytes(pcm[i:i + 2], "little", signed=True)) for i in range(0, len(pcm), 2)) > 3000


def test_status_shape(tmp_path):
    t, clock, _ = make(tmp_path, epoch(15, 0))
    t.set_alarm("7 pm")
    t.set_timer("5 minutes", "tea")
    s = t.status()
    assert s["ringing"] is False
    assert [r["kind"] for r in s["timers"]] == ["timer", "alarm"]
    assert s["timers"][0]["left"] == 300 and s["timers"][1]["at"] == "7 PM"
    json.dumps(s)


def test_tools_declared_for_both_backends():
    from reachy_gemini_dj.tool_schemas import INFO_TOOLS, robot_tool_specs
    specs = {s["name"]: s for s in robot_tool_specs()}
    for name in ("set_timer", "set_alarm", "list_timers", "cancel_timer", "stop_alarm"):
        assert name in specs
    assert specs["list_timers"]["kind"] == "info" and "list_timers" in INFO_TOOLS
    assert specs["set_timer"]["parameters"]["required"] == ["duration"]
    assert set(specs["set_timer"]["parameters"]["properties"]) == {"duration", "label"}
    assert set(specs["set_alarm"]["parameters"]["properties"]) == {"time", "label"}
    assert set(specs["cancel_timer"]["parameters"]["properties"]) == {"which"}
