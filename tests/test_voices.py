"""Voices tab: the brain's voice store, the brain's voice requests, and the app's link to them (fakes, no models)."""
import asyncio
import base64
import json
import shutil
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_brain import voices as V
from reachy_brain.config import BrainConfig
from reachy_brain.server import Brain

AUDIO = b"x" * 5000                                   # stands in for an uploaded mp3


def fake_convert(seconds=12.0):
    def convert(src, dst):
        assert src.read_bytes() == AUDIO
        dst.write_bytes(b"RIFF" + bytes(int(V.WAV_BYTES_PER_SECOND * seconds)))
    return convert


def fake_clone(wav, out):
    assert wav.exists()
    out.write_bytes(bytes(5000))


def store(tmp_path, seconds=12.0):
    return V.VoiceStore(tmp_path / "voices", convert=fake_convert(seconds))


def files(s):
    return sorted(p.name for p in s.root.iterdir())


# ----------------------------------------------------------------------------------------------------- the store
def test_create_use_and_delete_a_voice(tmp_path):
    s = store(tmp_path)
    meta = s.create("  Dan\n  Rock ", True, AUDIO, "dan.mp3", fake_clone)
    assert meta["name"] == "Dan Rock" and meta["consent"]["confirmed"] is True and meta["seconds"] == 12.0
    assert [v["id"] for v in s.list()] == [meta["id"]]
    # only the small voice file and its details are kept: not the recording, and no work folder is left behind
    assert files(s) == [f"{meta['id']}.json", f"{meta['id']}.safetensors"]
    assert (s.root / f"{meta['id']}.safetensors").stat().st_mode & 0o077 == 0         # owner only
    assert s.active() == V.BUILTIN_ID and s.spec() is None
    s.set_active(meta["id"])
    assert V.VoiceStore(s.root).active() == meta["id"]                                  # survives a restart
    assert s.spec() == s.root / f"{meta['id']}.safetensors"
    assert s.delete(meta["id"]) and s.list() == [] and s.active() == V.BUILTIN_ID      # back to the built-in voice
    assert files(s) == []
    assert not s.delete(meta["id"])


@pytest.mark.parametrize("consent", [False, None, "true", 1, "yes"])
def test_a_voice_is_never_made_without_the_consent_tick(tmp_path, consent):
    s = store(tmp_path)
    with pytest.raises(V.VoiceError, match="agreed"):
        s.create("Dan", consent, AUDIO, "dan.mp3", fake_clone)
    assert files(s) == []


def test_name_is_cleaned_and_required(tmp_path):
    s = store(tmp_path)
    assert V.clean_name("  a\x00b\tc  ") == "a b c"
    assert len(V.clean_name("x" * 100)) == V.NAME_MAX
    for bad in ("", "   ", None, "\n\t"):
        with pytest.raises(V.VoiceError, match="name"):
            s.create(bad, True, AUDIO, "a.mp3", fake_clone)


def test_bad_recordings_are_refused_and_leave_nothing(tmp_path):
    s = store(tmp_path)
    with pytest.raises(V.VoiceError, match="empty"):
        s.create("Dan", True, b"tiny", "a.mp3", fake_clone)
    with pytest.raises(V.VoiceError, match="too big"):
        s.create("Dan", True, bytes(V.MAX_UPLOAD_BYTES + 1), "a.mp3", fake_clone)
    with pytest.raises(V.VoiceError, match="seconds of sound"):
        store(tmp_path, seconds=1.0).create("Dan", True, AUDIO, "a.mp3", fake_clone)

    def unreadable(src, dst):
        raise V.VoiceError("I could not read that file as audio.")
    with pytest.raises(V.VoiceError, match="audio"):
        V.VoiceStore(s.root, convert=unreadable).create("Dan", True, AUDIO, "a.mp3", fake_clone)

    def broken_clone(wav, out):
        pass                                                                            # engine wrote nothing
    with pytest.raises(V.VoiceError, match="did not make"):
        s.create("Dan", True, AUDIO, "a.mp3", broken_clone)
    assert files(s) == []


def test_the_voice_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "MAX_VOICES", 2)
    s = store(tmp_path)
    s.create("A", True, AUDIO, "a.mp3", fake_clone)
    s.create("B", True, AUDIO, "a.mp3", fake_clone)
    with pytest.raises(V.VoiceError, match="Delete one first"):
        s.create("C", True, AUDIO, "a.mp3", fake_clone)


@pytest.mark.parametrize("vid", ["../x", "a/b", "", None, "..", "x.y", "A", "x" * 41, "x\x00", 5])
def test_an_id_cannot_leave_the_voices_folder(tmp_path, vid):
    s = store(tmp_path)
    (tmp_path / "x.safetensors").write_bytes(bytes(2000))                               # a file next to the folder
    assert s.path(vid) is None and not s.delete(vid)
    with pytest.raises(V.VoiceError):
        s.set_active(vid)


def test_a_voice_file_copied_in_by_hand_is_listed_without_a_consent_record(tmp_path):
    s = store(tmp_path)
    (s.root / "dan_voice.safetensors").write_bytes(bytes(2000))
    (s.root / "not-a-voice.txt").write_text("x")
    (s.root / "Bad Name.safetensors").write_bytes(bytes(2000))                          # not a safe id: ignored
    [v] = s.list()
    assert v["id"] == "dan_voice" and v["name"] == "Dan Voice" and v["consent"]["confirmed"] is False
    s.set_active("dan_voice")
    assert s.spec() == s.root / "dan_voice.safetensors"


def test_audio_upload_decoding():
    assert V.decode_audio(base64.b64encode(b"abc").decode()) == b"abc"
    with pytest.raises(V.VoiceError, match="damaged"):
        V.decode_audio("***not base64***")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ffmpeg_makes_a_mono_24k_wav_and_refuses_garbage(tmp_path):
    src = tmp_path / "in.wav"
    t = np.arange(48000 * 3) / 48000
    pcm = (np.sin(2 * np.pi * 220 * t) * 12000).astype("<i2")
    with wave.open(str(src), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000); f.writeframes(pcm.tobytes())
    dst = tmp_path / "out.wav"
    V.ffmpeg_to_wav(src, dst)
    with wave.open(str(dst), "rb") as f:
        assert (f.getnchannels(), f.getframerate(), f.getsampwidth()) == (1, 24000, 2) and abs(f.getnframes() / 24000 - 3) < 0.1
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"this is not audio at all" * 100)
    with pytest.raises(V.VoiceError, match="could not read"):
        V.ffmpeg_to_wav(junk, tmp_path / "junk.wav")


# ------------------------------------------------------------------------------------- the brain's voice requests
class FakeTTS:
    def __init__(self, can_clone=True):
        self.can_clone, self.voice, self.cloned = can_clone, "default", 0

    def clone(self, wav, out):
        self.cloned += 1
        fake_clone(wav, out)

    def set_voice(self, spec):
        self.voice = spec


def brain(tmp_path, can_clone=True):
    b = Brain(BrainConfig(), llm=object(), stt=object(), tts=FakeTTS(can_clone))
    b.voices = V.VoiceStore(tmp_path / "voices", convert=fake_convert())
    return b


def req(**kw):
    return {"rid": "r1", **kw}


def test_brain_voice_requests_end_to_end(tmp_path):
    b = brain(tmp_path)
    r = b.voice_op(req(type="voice_list"))
    assert r == {"type": "voice_result", "rid": "r1", "ok": True, "voices": [], "active": "default", "can_clone": True}
    r = b.voice_op(req(type="voice_create", name="Dan", consent=True, filename="dan.mp3",
                       audio_b64=base64.b64encode(AUDIO).decode()))
    assert r["ok"] and r["voice"]["name"] == "Dan" and b.tts.cloned == 1
    vid = r["voice"]["id"]
    r = b.voice_op(req(type="voice_use", id=vid))
    assert r["ok"] and r["active"] == vid and b.tts.voice == b.voices.path(vid)
    r = b.voice_op(req(type="voice_delete", id=vid))
    assert r["ok"] and r["active"] == "default" and r["voices"] == [] and b.tts.voice is None   # engine back on its own voice


def test_brain_refuses_cloning_without_the_model_or_consent(tmp_path):
    b = brain(tmp_path, can_clone=False)
    r = b.voice_op(req(type="voice_create", name="Dan", consent=True, filename="a.mp3", audio_b64=base64.b64encode(AUDIO).decode()))
    assert not r["ok"] and "Hugging Face" in r["error"] and b.tts.cloned == 0
    b = brain(tmp_path)
    r = b.voice_op(req(type="voice_create", name="Dan", consent=False, filename="a.mp3", audio_b64=base64.b64encode(AUDIO).decode()))
    assert not r["ok"] and "agreed" in r["error"] and b.tts.cloned == 0


def test_brain_voice_errors_never_raise(tmp_path):
    b = brain(tmp_path)
    assert not b.voice_op(req(type="voice_use", id="nope"))["ok"]
    assert not b.voice_op(req(type="voice_use", id="../etc"))["ok"]
    assert not b.voice_op(req(type="voice_delete", id="nope"))["ok"]
    assert not b.voice_op(req(type="voice_dance"))["ok"]
    assert not b.voice_op(req(type="voice_create", name="Dan", consent=True, audio_b64="***"))["ok"]
    b.voices = None                                                                       # still starting up
    assert "starting" in b.voice_op(req(type="voice_list"))["error"]


# ------------------------------------------------------------------------------- the app's link to the brain
def session():
    from reachy_gemini_dj.remote_session import RemoteSession
    return RemoteSession(SimpleNamespace(), lambda n, a: {}, lambda k, t: None)


def run_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def test_voice_call_waits_for_the_brains_answer():
    s, loop = session(), run_loop()
    sent = []

    class Brainy:
        async def send(self, data):
            m = json.loads(data); sent.append(m)
            loop.call_later(0.05, lambda: s._voice_waiters[m["rid"]].set_result({"ok": True, "voices": []}))
    s._loop, s._ws, s.connected = loop, Brainy(), True
    assert s.voice_call({"type": "voice_list"}, timeout=5) == {"ok": True, "voices": []}
    assert sent[0]["type"] == "voice_list" and sent[0]["rid"] and s._voice_waiters == {}
    loop.call_soon_threadsafe(loop.stop)


def test_voice_call_reports_a_missing_brain_a_silent_brain_and_a_dropped_link():
    s = session()
    assert s.voice_call({"type": "voice_list"}) == {"ok": False, "error": "The voice brain is not connected right now."}
    loop = run_loop()

    class Silent:
        async def send(self, data):
            pass
    s._loop, s._ws, s.connected = loop, Silent(), True
    r = s.voice_call({"type": "voice_list"}, timeout=0.2)
    assert not r["ok"] and "did not answer" in r["error"] and s._voice_waiters == {}
    results = []
    t = threading.Thread(target=lambda: results.append(s.voice_call({"type": "voice_list"}, timeout=5)))
    t.start()
    for _ in range(100):
        if s._voice_waiters:
            break
        threading.Event().wait(0.02)
    s._fail_voice_waiters("The brain disconnected.")
    t.join(5)
    assert results == [{"ok": False, "error": "The brain disconnected."}]
    loop.call_soon_threadsafe(loop.stop)


def test_a_voice_result_message_resolves_the_waiting_request():
    import concurrent.futures
    s = session()
    fut = concurrent.futures.Future()
    s._voice_waiters["abc"] = fut

    class WS:
        def __aiter__(self):
            async def gen():
                yield json.dumps({"type": "voice_result", "rid": "other", "ok": True})       # someone else's: ignored
                yield json.dumps({"type": "voice_result", "rid": "abc", "ok": True, "active": "default"})
            return gen()
    asyncio.run(s._recv_loop(WS()))
    assert fut.result(0) == {"ok": True, "active": "default"}
