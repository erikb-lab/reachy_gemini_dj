"""motor_health.py: Dynamixel 2.0 packets and the read / clear flow against a simulated motor bus."""
import pytest

from reachy_gemini_dj import motor_health as mh


def test_crc_matches_robotis_examples():
    # ROBOTIS e-Manual protocol 2.0 examples: Ping ID 1 and Reboot ID 1
    assert mh.packet(1, 0x01).hex() == "fffffd0001030001194e"
    assert mh.packet(1, 0x08).hex() == "fffffd00010300082f4e"


def status(mid, params=b"", err=0x00):
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, mid]) + (len(params) + 4).to_bytes(2, "little") + bytes([0x55, err]) + params
    return body + mh.crc16(body).to_bytes(2, "little")


def test_parse_status_checks_everything():
    assert mh.parse_status(status(18, b"\x20", err=0x80), 18) == b"\x20"      # alert bit alone is fine
    with pytest.raises(ValueError):
        mh.parse_status(status(17, b"\x20"), 18)                              # wrong motor
    bad = bytearray(status(18, b"\x20")); bad[-1] ^= 0xFF
    with pytest.raises(ValueError):
        mh.parse_status(bytes(bad), 18)                                       # checksum
    with pytest.raises(ValueError):
        mh.parse_status(status(18, err=0x07), 18)                             # instruction error
    with pytest.raises(ValueError):
        mh.parse_status(b"", 18)


class FakeBus:
    """Answers read / write / reboot packets like XL330s: hardware error at 70, PID at 80-85, voltage at 144."""
    def __init__(self, faults=None, volts=50, stuck=()):
        self.reg = {mid: {"err": 0, "pid": bytes([0, 0, 0, 0, 200, 0])} for mid in mh.MOTORS.values()}
        for mid, e in (faults or {}).items():
            self.reg[mid]["err"] = e
        self.volts, self.stuck, self.out, self.log = volts, set(stuck), [], []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def send(self, pkt):
        mid, inst = pkt[4], pkt[7]
        params = pkt[8:-2]
        r = self.reg[mid]
        self.log.append((mid, inst))
        alert = 0x80 if r["err"] else 0
        if inst == mh.INST_READ:
            addr = int.from_bytes(params[0:2], "little")
            data = {70: bytes([r["err"]]), 80: r["pid"], 144: self.volts.to_bytes(2, "little")}[addr]
            self.out.append(status(mid, data, alert))
        elif inst == mh.INST_WRITE:
            if int.from_bytes(params[0:2], "little") == 80:
                r["pid"] = bytes(params[2:])
            self.out.append(status(mid, b"", alert))
        elif inst == mh.INST_REBOOT:
            self.out.append(status(mid, b"", alert))
            if mid not in self.stuck:
                r["err"] = 0
            r["pid"] = bytes([0, 0, 0, 0, 0x90, 0x01])     # defaults after a reboot (P=400)
    def recv(self, timeout=None):
        return self.out.pop(0)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(mh, "REBOOT_WAIT_S", 0)


def test_read_reports_overload_and_ignores_normal_voltage_flag():
    bus = FakeBus(faults={18: 0x20, 17: 0x01}, volts=50)
    res = mh.MotorHealth(connect=lambda: bus).read()
    assert res["faults"] == ["left_antenna"]
    assert res["motors"]["left_antenna"]["errors"] == ["Overload"]
    assert res["motors"]["right_antenna"]["errors"] == []                     # 5.0 V: voltage flag ignored


def test_read_keeps_real_voltage_problem():
    bus = FakeBus(faults={17: 0x01}, volts=90)
    assert mh.MotorHealth(connect=lambda: bus).read()["faults"] == ["right_antenna"]


def test_clear_reboots_only_faulty_motor_and_restores_pid():
    bus = FakeBus(faults={18: 0x20})
    res = mh.MotorHealth(connect=lambda: bus).clear()
    assert res["faults"] == [] and res["cleared"]["left_antenna"]["ok"]
    assert res["cleared"]["left_antenna"]["before"] == ["Overload"]
    assert bus.reg[18]["pid"][4:] == bytes([200, 0])                          # P gain put back to the daemon's 200
    reboots = [m for m, i in bus.log if i == mh.INST_REBOOT]
    assert reboots == [18]


def test_clear_reports_motor_that_stays_faulty():
    bus = FakeBus(faults={18: 0x20}, stuck={18})
    res = mh.MotorHealth(connect=lambda: bus).clear()
    assert res["faults"] == ["left_antenna"] and res["cleared"]["left_antenna"]["after"] == ["Overload"]


def test_clear_with_nothing_wrong_does_nothing():
    bus = FakeBus()
    res = mh.MotorHealth(connect=lambda: bus).clear()
    assert res == {"cleared": {}, "faults": []}
    assert all(i == mh.INST_READ for _, i in bus.log)
