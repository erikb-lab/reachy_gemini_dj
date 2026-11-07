"""Offline tests: multi-song queue, prefetch, autoplay, queue tools, favorites, volume steps, seek.
yt-dlp and ffmpeg/ffplay are mocked (same fakes as test_music_playback)."""
import json
import re
import time

import pytest

from reachy_gemini_dj import favorites as fav_mod
from reachy_gemini_dj import music as music_mod
from reachy_gemini_dj.music import parse_seek
from test_music_playback import FakeProc, env, wait_for  # noqa: F401  (env is a fixture)


def make_search(total=12, long_ids=(), dup_of=None):
    """Fake yt-dlp: ytsearchN:q -> flat entries; anything else -> a resolvable stream."""
    def info(q):
        m = re.match(r"ytsearch(\d+):(.*)", q)
        if m:
            n = int(m.group(1))
            ents = [{"id": f"id{i}", "title": f"Song {i}", "uploader": "Artist", "duration": 4000 if i in long_ids else 200,
                     "url": f"https://y/watch?v=id{i}"} for i in range(min(n, total))]
            if dup_of is not None and ents:
                ents.insert(1, dict(ents[0]))
            return {"entries": ents}
        mm = re.search(r"watch\?v=(id\d+)", q)
        title = "Song " + mm.group(1)[2:] if mm else q
        return {"url": "http://s/" + q, "title": title, "uploader": "Artist", "duration": 200, "id": mm.group(1) if mm else ""}
    return info


@pytest.fixture
def favs(tmp_path, monkeypatch):
    monkeypatch.setattr(fav_mod, "FAV_FILE", tmp_path / "favorites.json")
    return tmp_path / "favorites.json"


def titles(p):
    return [i["title"] for i in p.get_status()["queue_items"]]


def test_count_queues_different_tracks(env):
    env.info = make_search(long_ids={1}, dup_of=0)
    p = env.make()
    r = p.play("Daft Punk songs", count=3)
    assert r["say"] == "Okay, playing 3 Daft Punk songs."
    assert wait_for(lambda: p.get_status()["track"] == "Song 0")
    assert wait_for(lambda: len(p.get_status()["queue_items"]) == 2)
    # duplicate skipped, 4000s video skipped
    assert titles(p) == ["Song 2", "Song 3"]
    assert any(c[0].startswith("ytsearch9:Daft Punk") for c in env.calls)


def test_long_videos_allowed_for_mix_queries(env):
    env.info = make_search(long_ids={0})
    p = env.make()
    p.play("chill mix", count=2)
    assert wait_for(lambda: p.get_status()["track"] == "Song 0")


def test_count_clamped_and_say_when_busy(env):
    env.info = make_search()
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    r = p.play("rock", count=99)
    assert r["say"] == "Okay, queuing 10 rock songs."
    assert wait_for(lambda: p.get_status()["queued"] == 10)
    assert p.play("x", count=3, when="next")["say"] == "Okay, queuing 3 x songs next."


def test_when_semantics(env):
    env.info = make_search()
    p = env.make()
    assert p.play("one")["say"] == "Okay, playing one."
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    assert p.play("two")["say"] == "Okay, queuing that up."
    p.play("three")
    assert titles(p) == ["two", "three"]
    assert p.play("urgent", when="next")["say"] == "Okay, that's up next."
    assert titles(p) == ["urgent", "two", "three"]
    first = env.procs[0]
    r = p.play("now-song", when="now")
    assert r["say"] == "Playing now-song now."
    assert wait_for(lambda: p.get_status()["track"] == "now-song")
    assert "TERM" in first.signals                      # current track was stopped
    assert titles(p) == ["urgent", "two", "three"]      # rest of the queue kept


def test_prefetch_next_while_playing(env):
    env.info = make_search()
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    p.play("b")
    assert wait_for(lambda: ("b", False) in env.calls)          # resolved before it is its turn
    n = len(env.calls)
    p.skip()
    assert wait_for(lambda: p.get_status()["track"] == "b")
    assert [c[0] for c in env.calls].count("b") == 1 and len(env.calls) == n   # no second lookup


def test_stale_prefetch_is_re_resolved(env):
    env.info = make_search()
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    p.play("b")
    assert wait_for(lambda: ("b", False) in env.calls)
    with p._lock:
        p._queue[0]["info_ts"] = time.time() - 5 * 3600
    p.skip()
    assert wait_for(lambda: p.get_status()["track"] == "b")
    assert [c[0] for c in env.calls].count("b") == 2


def test_autoplay_continues_without_repeats(env):
    env.info = make_search(total=40)
    env.next_proc = lambda: FakeProc(run_for=0.05)
    p = env.make()
    p.autoplay = True
    seen = []
    p.play("daft punk")
    def collect():
        t = p.get_status()["track"]
        if t and (not seen or seen[-1] != t):
            seen.append(t)
        return len(seen) >= 6
    assert wait_for(collect, 8)
    assert len(set(seen)) == len(seen)
    assert any(c[0].startswith("ytsearch10:daft punk") for c in env.calls)


def test_autoplay_off_stops_and_stop_cancels(env):
    env.info = make_search()
    env.next_proc = lambda: FakeProc(run_for=0.05)
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "idle" and ("a", False) in env.calls)
    time.sleep(0.2)
    assert not any(c[0].startswith("ytsearch") for c in env.calls)   # autoplay off by default in MusicPlayer
    env.next_proc = lambda: FakeProc()
    p.autoplay = True
    p.play("b")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    p.stop()
    time.sleep(0.3)                          # let any in-flight prefetch finish, then nothing may start
    calls = list(env.calls)
    time.sleep(0.4)
    assert p.get_status()["state"] == "idle" and env.calls == calls and len(env.procs) == 2


def test_queue_actions_and_say(env):
    env.info = make_search()
    p = env.make()
    assert p.queue_action("now_playing")["say"] == "Nothing is playing."
    assert p.queue_action("list")["say"] == "The queue is empty."
    p.play("Get Lucky")
    assert wait_for(lambda: p.get_status()["track"] == "Get Lucky")
    assert p.queue_action("now_playing")["say"] == "This is Get Lucky by Artist."
    for n in "abcde":
        p.play(n)
    assert p.queue_action("list")["say"] == "Up next: a, then b, and 3 more."
    assert p.queue_action("remove", "2")["say"] == "Removed b."
    assert p.queue_action("remove", "last")["say"] == "Removed e."
    assert p.queue_action("remove", "next")["say"] == "Removed a."
    assert p.queue_action("remove", "c")["say"] == "Removed c."
    assert "error" in p.queue_action("remove", "zzz")
    assert p.queue_action("list")["say"] == "Up next: d."
    assert p.queue_action("shuffle")["say"] == "There's nothing to shuffle."
    p.play("x"); p.play("y"); p.play("z")
    assert p.queue_action("shuffle")["say"] == "Shuffled the queue."
    assert sorted(titles(p)) == ["d", "x", "y", "z"]
    assert p.queue_action("autoplay_on")["autoplay"] is True and p.get_status()["autoplay"] is True
    assert p.queue_action("autoplay_off")["say"] == "Okay, I'll stop when the queue ends."
    assert p.queue_action("clear")["say"] == "Cleared the queue."
    assert p.get_status()["queued"] == 0
    assert "error" in p.queue_action("bogus")


def test_remove_and_move_next_by_id(env):
    env.info = make_search()
    p = env.make()
    p.play("one")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    p.play("two"); p.play("three")
    ids = {i["title"]: i["id"] for i in p.get_status()["queue_items"]}
    p.move_next(ids["three"])
    assert titles(p) == ["three", "two"]
    p.remove_item(ids["three"])
    assert titles(p) == ["two"]
    assert "error" in p.remove_item(999)


def test_favorites_persistence(env, favs):
    env.info = make_search()
    p = env.make()
    assert "error" in p.favorite("save")                      # nothing playing
    p.play("https://y/watch?v=id7")
    assert wait_for(lambda: p.get_status()["track"] == "Song 7")
    assert p.favorite("save")["say"] == "Saved Song 7 to your favorites."
    assert "already" in p.favorite("save")["say"]             # deduped
    data = json.loads(favs.read_text())
    assert len(data) == 1 and data[0]["id"] == "id7" and data[0]["title"] == "Song 7"
    assert not favs.with_suffix(".tmp").exists()               # atomic write left nothing behind
    for i in range(210):
        fav_mod.add(f"T{i}", video_id=f"v{i}")
    assert len(fav_mod.load()) == fav_mod.MAX_FAVORITES
    assert p.favorite("list")["say"].startswith("You have 200 favorites, including T209")
    assert fav_mod.remove("v209") and len(fav_mod.load()) == 199


def test_favorites_play_queues_shuffled(env, favs):
    env.info = make_search()
    p = env.make()
    assert "error" in p.favorite("play")
    for i in range(3):
        fav_mod.add(f"F{i}", video_id=f"f{i}", url=f"https://y/watch?v=id{i}")
    assert p.favorite("play")["say"] == "Playing your 3 favorites."
    assert wait_for(lambda: p.get_status()["track"] != "" and p.get_status()["queued"] == 2)


def test_adjust_volume_clamps(env):
    p = env.make()                       # volume 0.5
    r = p.adjust_volume("up")
    assert r["say"] == "Louder." and r["percent"] == 65
    assert p.adjust_volume("down")["percent"] == 50
    p.set_volume(2.45)
    r = p.adjust_volume("up")
    assert r["volume"] == 2.5 and r["say"] == "Louder."
    r = p.adjust_volume("up")
    assert r["say"] == "That's as loud as I go."
    p.set_volume(0.1)
    assert p.adjust_volume("down")["volume"] == 0.0
    assert "error" in p.adjust_volume("sideways")


def test_speakable_errors():
    r = music_mod.speakable({"error": "x", "say": "Nothing is playing."})
    assert r == {"error": "Nothing is playing."}


# ---------------- seek ----------------
@pytest.mark.parametrize("text,expected", [
    ("1:30", ("abs", 90)), ("2 minutes", ("abs", 120)), ("90 seconds", ("abs", 90)),
    ("1 minute 15", ("abs", 75)), ("1 minute 15 seconds", ("abs", 75)), ("a minute", ("abs", 60)),
    ("half a minute", ("abs", 30)), ("1:02:03", ("abs", 3723)), ("90", ("abs", 90)),
    ("+30", ("rel", 30)), ("-10", ("rel", -10)), ("forward 30 seconds", ("rel", 30)),
    ("ahead a minute", ("rel", 60)), ("back 15 seconds", ("rel", -15)), ("rewind 10", ("rel", -10)),
    ("go back thirty seconds", ("rel", -30)),
    ("start", ("abs", 0)), ("the beginning", ("abs", 0)), ("start over", ("abs", 0)),
])
def test_parse_seek(text, expected):
    assert parse_seek(text) == (expected[0], float(expected[1]))


def test_parse_seek_garbage():
    assert parse_seek("") is None and parse_seek("somewhere nice") is None


def start_playing(env, **info):
    base = {"url": "http://s/x", "title": "T", "duration": 200}
    base.update(info)
    env.info = lambda q: dict(base)
    p = env.make()
    p.play("x")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    return p


def test_seek_passes_ss_and_reuses_stream(env):
    p = start_playing(env)
    first = env.procs[0]
    r = p.seek("1:30")
    assert r["say"] == "Skipping to 1:30."
    assert wait_for(lambda: len(env.decoders) == 2 and p.get_status()["state"] == "playing")
    cmd = env.decoders[1].cmd
    assert cmd[cmd.index("-ss") + 1] == "90.00" and cmd.index("-ss") < cmd.index("-i")
    assert "-ss" not in env.decoders[0].cmd
    assert env.calls == [("x", False)]                       # no re-resolve
    assert "TERM" in first.signals
    assert p.get_status()["elapsed"] >= 90 and p.get_status()["track"] == "T"


def test_seek_relative_and_start(env):
    p = start_playing(env)
    assert p.seek("+30")["say"] == "Skipping ahead 30 seconds."
    assert wait_for(lambda: len(env.decoders) == 2 and p.get_status()["state"] == "playing")
    assert p.seek("back 10 seconds")["say"] == "Going back 10 seconds."
    assert wait_for(lambda: len(env.decoders) == 3 and p.get_status()["state"] == "playing")
    cmd = env.decoders[2].cmd
    assert 19.0 <= float(cmd[cmd.index("-ss") + 1]) <= 21.0
    assert p.seek("start")["say"] == "Starting the song over."
    assert wait_for(lambda: len(env.decoders) == 4 and p.get_status()["state"] == "playing")
    assert "-ss" not in env.decoders[3].cmd
    assert p.seek("rewind 500")["say"] == "Starting the song over."


def test_seek_past_end_skips_to_next(env):
    env.info = lambda q: {"url": "http://s/" + q, "title": q, "duration": 100}
    p = env.make()
    p.play("a"); p.play("b")
    assert wait_for(lambda: p.get_status()["track"] == "a")
    r = p.seek("5:00")
    assert r["say"] == "That's past the end, skipping to the next song."
    assert wait_for(lambda: p.get_status()["track"] == "b")


def test_seek_refuses_live_and_idle(env):
    p = env.make()
    assert p.seek("1:00")["say"] == "Nothing is playing."
    env.info = lambda q: {"url": "http://live", "title": "L", "duration": None, "is_live": True}
    p.play("live")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    assert p.seek("1:00")["say"] == "I can't skip around in a live stream."
    assert len(env.decoders) == 1


def test_seek_while_paused_stays_paused(env):
    p = start_playing(env)
    p.pause()
    assert p.seek("1:00")["say"] == "Skipping to 1:00."
    assert wait_for(lambda: len(env.decoders) == 2 and p.get_status()["state"] == "paused")
    assert p.get_status()["elapsed"] == pytest.approx(60, abs=0.5)
    assert "error" in p.seek("garbage words")
