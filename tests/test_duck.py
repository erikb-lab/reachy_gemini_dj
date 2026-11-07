"""Music ducking policy: push-to-talk ducks only while talk is held or the robot speaks (fakes only)."""
import numpy as np

from test_ptt import LOUD, make_pipe, run_loop

SPEECH = np.zeros(2400, dtype="<i2").tobytes()


def kinds(p):
    return [d[0] for d in p.music.ducks]


def test_ptt_crowd_noise_never_ducks():
    p, _, stop = make_pipe("push_to_talk", [LOUD.copy() for _ in range(15)])
    run_loop(p, stop)
    assert p._user_speaking                              # VAD still tracks it
    assert "duck" not in kinds(p)


def test_always_mode_vad_still_ducks_and_unducks():
    p, _, stop = make_pipe("always", [LOUD.copy() for _ in range(5)])
    run_loop(p, stop)
    assert kinds(p)[0] == "duck" and p._user_speaking
    p2, _, stop2 = make_pipe("always", [])
    p2._user_speaking, p2._last_user_speech = True, 0.0   # hangover long over
    p2.media.frames.append(np.zeros(320, dtype="<i2"))
    run_loop(p2, stop2)
    assert kinds(p2) == ["reset"]


def test_ptt_press_ducks_release_unducks():
    p, _, _ = make_pipe("push_to_talk")
    p.ptt_set(True)
    assert kinds(p) == ["duck"]
    p.ptt_set(False)
    assert kinds(p) == ["duck", "reset"]


def test_release_keeps_duck_while_robot_speaks():
    p, _, _ = make_pipe("push_to_talk")
    p.ptt_set(True)
    p.play_voice(SPEECH)                                 # robot speaking
    p._is_robot_speaking = True
    p.ptt_set(False)
    assert kinds(p)[-1] == "duck"


def test_robot_finish_unducks_unless_talk_held():
    p, _, _ = make_pipe("push_to_talk")
    p.play_voice(SPEECH)
    p._ptt.press()
    p._is_robot_speaking = False
    p._update_duck()
    assert kinds(p)[-1] == "duck"                        # talk held
    p._ptt.release()
    p._update_duck()
    assert kinds(p)[-1] == "reset"


def test_clear_voice_ptt_unducks():
    p, _, _ = make_pipe("push_to_talk")
    p.play_voice(SPEECH)
    p._user_speaking = True                              # VAD noise must not matter in PTT
    p.clear_voice()
    assert kinds(p)[-1] == "reset"


def test_mode_switch_while_ducked_does_not_stick():
    p, _, stop = make_pipe("push_to_talk")
    p.ptt_set(True)
    assert kinds(p)[-1] == "duck"
    p.config.listen_mode = "always"
    p.media.frames.extend(np.zeros(320, dtype="<i2") for _ in range(5))
    run_loop(p, stop)
    assert kinds(p)[-1] == "reset"
