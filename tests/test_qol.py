"""Conversation log, brain chip, quiet mode and the greeting (pure logic, fake clock, no robot)."""
import datetime

from reachy_gemini_dj.qol import (ConversationLog, Greeter, QuietMode, brain_status, clamp_minutes, fmt_clock)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def adv(self, s):
        self.t += s


def make_log(backend="local"):
    c = Clock()
    return ConversationLog(backend=lambda: backend, clock=c, fmt=lambda ts: f"t{int(ts)}"), c


# ------------------------------------------------------------- conversation log: local brain ---
def test_local_turn_joins_clauses_and_measures_latency():
    log, c = make_log("local")
    log.heard_utterance("what's the weather")
    c.adv(3.24)
    log.said("It is sunny.")
    c.adv(1)
    log.said("High of 20.")
    log.turn_done()
    (t,) = log.listing()
    assert t["heard"] == "what's the weather"
    assert t["said"] == "It is sunny. High of 20."
    assert t["latency_s"] == 3.2
    assert t["backend"] == "local" and t["open"] is False
    assert log.last_latency("local") == 3.2 and log.last_latency("gemini") is None


def test_local_turns_newest_first_and_capped():
    log, c = make_log()
    for i in range(25):
        log.heard_utterance(f"q{i}")
        c.adv(1)
        log.said(f"a{i}")
        log.turn_done()
        c.adv(10)
    items = log.listing()
    assert len(items) == 20
    assert items[0]["heard"] == "q24" and items[-1]["heard"] == "q5"


def test_new_utterance_closes_unanswered_turn():
    log, c = make_log()
    log.heard_utterance("hello?")
    c.adv(5)
    log.heard_utterance("anyone there")
    items = log.listing()
    assert [t["heard"] for t in items] == ["anyone there", "hello?"]
    assert items[1]["said"] == "" and items[1]["latency_s"] is None


def test_announcement_without_heard_is_its_own_turn_without_latency():
    log, c = make_log()
    log.said("Your timer is done.")
    (t,) = log.listing()
    assert t["heard"] == "" and t["said"] == "Your timer is done." and t["latency_s"] is None


def test_late_transcript_piece_joins_the_turn_that_just_ended():
    log, c = make_log("gemini")
    log.heard_fragment("hi")
    c.adv(1)
    log.said("Hello")
    log.turn_done()
    c.adv(0.5)
    log.said(" there")
    items = log.listing()
    assert len(items) == 1 and items[0]["said"] == "Hello there"
    c.adv(5)
    log.said("Reminder!")
    assert len(log.listing()) == 2


# ------------------------------------------------------------- conversation log: Gemini Live ---
def test_gemini_fragments_group_into_one_turn():
    log, c = make_log("gemini")
    for piece in ("play ", "some ", "jazz"):
        log.heard_fragment(piece)
        c.adv(0.4)
    c.adv(1.2)
    log.said("Sure,")
    log.said(" playing jazz")
    log.said(" now.")
    log.turn_done()
    (t,) = log.listing()
    assert t["heard"] == "play some jazz"
    assert t["said"] == "Sure, playing jazz now."
    assert t["latency_s"] == 1.6
    assert t["backend"] == "gemini"


def test_gemini_heard_after_reply_started_is_a_new_turn():
    log, c = make_log("gemini")
    log.heard_fragment("one")
    c.adv(1)
    log.said("answer one")
    c.adv(1)
    log.heard_fragment("two")        # no turn_complete arrived, but Reachy was already answering
    items = log.listing()
    assert [t["heard"] for t in items] == ["two", "one"]


def test_gemini_long_pause_without_reply_starts_new_turn():
    log, c = make_log("gemini")
    log.heard_fragment("first")
    c.adv(20)
    log.heard_fragment("second")
    assert [t["heard"] for t in log.listing()] == ["second", "first"]


def test_blank_text_is_ignored():
    log, _ = make_log()
    log.heard_fragment("  ")
    log.said("")
    log.heard_utterance("")
    assert log.listing() == []


# ------------------------------------------------------------- brain chip ---
def st(**kw):
    base = {"mode": "auto", "active": "local", "reason": "", "connected": True}
    base.update(kw)
    return base


def test_chip_local_states():
    assert brain_status(st(), "listening", None)["chip"]["text"] == "Brain: ready"
    assert brain_status(st(), "loading", None)["chip"] == {"text": "Brain: warming up…", "level": "warn"}
    assert brain_status(st(), "thinking", None)["chip"]["text"] == "Brain: thinking"
    assert brain_status(st(connected=False), "connecting", None)["chip"]["level"] == "warn"


def test_chip_shows_last_latency_and_exposes_fields():
    b = brain_status(st(), "listening", 3.24)
    assert b["chip"]["text"] == "Brain: ready · last answer 3.2 s"
    assert b["state"] == "listening" and b["last_latency_s"] == 3.24 and b["mode"] == "auto"


def test_chip_offline_variants():
    assert brain_status(st(active="gemini", connected=False), "error", None)["chip"]["text"] == "Brain: offline — using Gemini"
    assert "Gemini only" in brain_status(st(active="gemini", mode="gemini", connected=False), "stopped", None)["chip"]["text"]
    off = brain_status(st(active="none", reason="local brain unreachable", connected=False), "error", None)
    assert off["chip"] == {"text": "Brain: offline", "level": "bad"}
    assert brain_status(st(active="none", connected=False), "starting", None)["chip"]["text"] == "Brain: starting…"


# ------------------------------------------------------------- quiet mode ---
class Rig:
    def __init__(self):
        self.clock = Clock(1_000_000.0)
        self.muted = []
        self.said = []
        self.events = []
        self.motion = FakeMotion()
        self.q = QuietMode(set_muted=self.muted.append, say=self.said.append, motion=self.motion,
                           on_event=lambda k, t: self.events.append((k, t)), clock=self.clock,
                           fmt=lambda ts: f"at{int(ts)}")


class FakeMotion:
    def __init__(self):
        self.calls = []

    def emotion(self, name):
        self.calls.append(("emotion", name))

    def move_head(self, d):
        self.calls.append(("head", d))


def test_quiet_enter_mutes_rests_and_returns_say():
    r = Rig()
    res = r.q.enter(30)
    assert res["say"] == "Okay, I'll stay quiet for 30 minutes. Press talk to wake me."
    assert r.muted == [True] and r.q.active
    assert r.motion.calls == [("emotion", "sleepy")]
    s = r.q.status()
    assert s["active"] and s["minutes_left"] == 30 and s["until_text"] == f"at{int(r.clock.t + 1800)}"
    assert r.said == []                       # the say string is returned to the brain, not spoken twice


def test_quiet_auto_wakes_with_fake_clock():
    r = Rig()
    r.q.enter(2)
    r.clock.adv(119)
    r.q.tick()
    assert r.q.active and r.said == []
    r.clock.adv(2)
    r.q.tick()
    assert not r.q.active
    assert r.muted == [True, False]
    assert r.said == ["I'm back."]
    assert r.motion.calls[-1] == ("head", "front")
    assert r.q.status() == {"active": False, "until": 0, "until_text": "", "minutes_left": 0}


def test_quiet_talk_press_wakes_and_only_when_active():
    r = Rig()
    r.q.on_talk()                              # nothing to wake
    assert r.said == [] and r.muted == []
    r.q.enter(10)
    r.q.on_talk()
    assert not r.q.active and r.said == ["I'm back."] and r.muted == [True, False]


def test_quiet_wake_without_say_and_double_wake():
    r = Rig()
    r.q.enter(5)
    assert r.q.wake(say=False) is True
    assert r.q.wake() is False
    assert r.said == [] and r.muted == [True, False]


def test_quiet_reenter_extends_without_resting_again():
    r = Rig()
    r.q.enter(5)
    r.clock.adv(60)
    r.q.enter(10)
    assert r.motion.calls == [("emotion", "sleepy")]
    assert r.q.status()["minutes_left"] == 10


def test_minutes_cap_and_parsing():
    assert clamp_minutes(0) == 1
    assert clamp_minutes(-5) == 1
    assert clamp_minutes(100000) == 480
    assert clamp_minutes("45") == 45
    assert clamp_minutes(None) == 30
    assert clamp_minutes("soon") == 30
    r = Rig()
    assert r.q.enter(1)["say"].startswith("Okay, I'll stay quiet for 1 minute.")
    assert r.q.enter(9999)["minutes"] == 480


def test_quiet_survives_motion_failures():
    r = Rig()

    def boom(*a):
        raise RuntimeError("no motors")
    r.motion.emotion = boom
    r.motion.move_head = boom
    r.q.enter(1)
    assert r.q.wake()
    assert r.said == ["I'm back."]


def test_quiet_without_motion():
    r = Rig()
    r.q.motion = None
    r.q.enter(1)
    assert r.q.wake()


def test_fmt_clock_uses_owner_timezone():
    from reachy_gemini_dj import timers
    ts = datetime.datetime(2026, 11, 6, 22, 45, tzinfo=datetime.timezone.utc).timestamp()
    old = timers._TZ
    try:
        assert timers.set_timezone("America/Chicago")
        assert fmt_clock(ts) == "4:45 PM"
        assert timers.set_timezone("Europe/London")
        assert fmt_clock(ts) == "10:45 PM"
    finally:
        timers._TZ = old


# ------------------------------------------------------------- greeting ---
def test_greeting_once_per_start():
    said = []
    g = Greeter(say=said.append)
    assert g.ready("local") is True
    assert g.ready("local") is False           # brain reconnects
    assert g.ready("gemini") is False
    assert said == ["I'm ready."]


def test_greeting_respects_setting_and_quiet_and_still_counts_as_done():
    said = []
    on = {"v": False}
    g = Greeter(say=said.append, enabled=lambda: on["v"])
    assert g.ready() is False
    on["v"] = True
    assert g.ready() is False and said == []   # the one chance has passed
    g2 = Greeter(say=said.append, is_quiet=lambda: True)
    assert g2.ready() is False and said == []


def test_greeting_failure_does_not_raise():
    def bad(t):
        raise RuntimeError("no voice")
    assert Greeter(say=bad).ready() is False
