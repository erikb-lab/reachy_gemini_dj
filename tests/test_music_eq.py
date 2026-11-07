"""Music EQ: ffmpeg filter construction, set_eq debounce/restart behaviour, pump pacing."""
import io
import time

from reachy_gemini_dj import gain as g
from reachy_gemini_dj import music as music_mod

# reuse the fake yt_dlp / ffmpeg / ffplay harness
from test_music_playback import env, wait_for  # noqa: F401


def test_flat_eq_adds_no_filter():
    plain = g.decoder_command("http://x/a", {"UA": "z"}, 5.0)
    assert g.decoder_command("http://x/a", {"UA": "z"}, 5.0, (0, 0, 0)) == plain
    assert g.decoder_command("http://x/a", {"UA": "z"}, 5.0, None) == plain
    assert "-af" not in plain


def test_eq_filter_string_and_position():
    cmd = g.decoder_command("http://x/a", None, 0.0, (3, -2, 6))
    af = cmd[cmd.index("-af") + 1]
    assert af == ("bass=g=3:f=100,equalizer=f=1000:t=o:w=1:g=-2,"
                  "treble=g=6:f=6000,alimiter=limit=0.95:level=disabled")
    assert cmd.index("-i") < cmd.index("-af") < cmd.index("pipe:1")


def test_cut_only_has_no_limiter_and_clamps():
    assert g.eq_filter((-4, 0, 0)) == "bass=g=-4:f=100"
    assert g.eq_filter((99, 0, 0)).startswith("bass=g=12:")
    assert g.eq_filter((float("nan"), "x", None)) == ""


def test_set_eq_clamps_and_idle_does_not_restart(env):
    p = env.make()
    p.set_eq(50, -50, 0.04)
    assert p._eq == (12.0, -12.0, 0.04)
    assert p._eq_timer is None and env.decoders == []


def test_playing_eq_change_restarts_once_debounced(env, monkeypatch):
    monkeypatch.setattr(music_mod, "EQ_DEBOUNCE_S", 0.15)
    p = env.make()
    p.play("song")
    assert wait_for(lambda: p.get_status()["state"] == "playing" and len(env.decoders) == 1)
    for v in (1, 2, 3, 4, 5):                # a slider drag
        p.set_eq(v, 0, 0)
    p.set_eq(5, 0, 0)                        # unchanged: no extra timer
    assert len(env.decoders) == 1            # nothing yet (debounced)
    assert wait_for(lambda: len(env.decoders) == 2)
    cmd = env.decoders[1].cmd
    assert "bass=g=5:f=100" in cmd[cmd.index("-af") + 1]
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    time.sleep(0.4)
    assert len(env.decoders) == 2            # exactly one restart


def test_paused_stays_paused_after_eq(env, monkeypatch):
    monkeypatch.setattr(music_mod, "EQ_DEBOUNCE_S", 0.05)
    p = env.make()
    p.play("song")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    p.pause()
    assert wait_for(lambda: p.get_status()["state"] == "paused")
    p.set_eq(0, 0, 4)
    assert wait_for(lambda: len(env.decoders) == 2)
    assert wait_for(lambda: p.get_status()["state"] == "paused" and len(env.procs) == 2)


def test_eq_applies_to_next_song(env):
    p = env.make()
    p.set_eq(0, 3, 0)
    p.play("song")
    assert wait_for(lambda: len(env.decoders) == 1)
    cmd = env.decoders[0].cmd
    assert "equalizer=f=1000" in cmd[cmd.index("-af") + 1]


def test_pump_pacing_bounds_lead():
    sleeps, now = [], [0.0]

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    blk = b"\x00" * (g.SAMPLE_RATE * 4 // 50)       # 20 ms
    pump = g.GainPump(io.BytesIO(blk * 50), io.BytesIO(), lambda: 1.0, max_lead_s=0.1,
                      clock=lambda: now[0], sleep=fake_sleep)
    pump.run()
    # 1 s of audio; the clock only advances via sleeps, so it ends ~max_lead short of 1 s
    assert abs(now[0] - 0.9) < 0.03
    # after a stall (clock jumps ahead) it re-anchors instead of bursting
    sleeps.clear()
    p2 = g.GainPump(io.BytesIO(b""), io.BytesIO(), lambda: 1.0, max_lead_s=0.1,
                    clock=lambda: now[0], sleep=fake_sleep)
    p2._pace(len(blk))
    now[0] += 50
    p2._pace(len(blk))
    p2._pace(len(blk))
    assert not sleeps
