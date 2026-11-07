"""Motor health: read each motor's hardware-error flags and clear them without a power cycle.

Reachy Mini's motors are Dynamixel XL330s. When one overloads (for example an antenna pushed against a stop) it
latches "Overload Error" and turns its torque off; only a power cycle or a Dynamixel *Reboot* instruction clears it.
Pollen's daemon has no button for that, but it does expose the motor bus as a raw-packet websocket
(``/api/move/ws/raw/write``), so this module talks Dynamixel protocol 2.0 through it:

* read Hardware Error Status (address 70) per motor, ignoring the input-voltage flag the way the daemon does when the
  measured voltage (address 144) is normal;
* clear = save the PID gains (addresses 80-85, RAM, lost on reboot), send Reboot, wait, write the gains back, check
  again. The caller then re-enables torque with ``enable_motors()``, which holds the motors where they are.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

RAW_URL = "ws://127.0.0.1:8000/api/move/ws/raw/write"
MOTORS = {"body_rotation": 10, "stewart_1": 11, "stewart_2": 12, "stewart_3": 13, "stewart_4": 14,
          "stewart_5": 15, "stewart_6": 16, "right_antenna": 17, "left_antenna": 18}
LABELS = {"body_rotation": "Body", "right_antenna": "Right antenna", "left_antenna": "Left antenna",
          **{f"stewart_{i}": f"Head motor {i}" for i in range(1, 7)}}
ERR_BITS = {0: "Input voltage", 2: "Overheating", 4: "Electrical shock", 5: "Overload"}
ADDR_HW_ERROR, ADDR_PID, ADDR_VOLTAGE = 70, 80, 144
INST_READ, INST_WRITE, INST_REBOOT, INST_STATUS = 0x02, 0x03, 0x08, 0x55
MAX_VOLTAGE = 7.8          # the daemon's own threshold for a real input-voltage problem
REBOOT_WAIT_S = 0.8


def crc16(data: bytes) -> int:
    """Dynamixel protocol 2.0 CRC (CRC-16/BUYPASS: poly 0x8005, init 0)."""
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def packet(motor_id: int, inst: int, params: bytes = b"") -> bytes:
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, motor_id]) + (len(params) + 3).to_bytes(2, "little") + bytes([inst]) + params
    return body + crc16(body).to_bytes(2, "little")


def read_packet(motor_id: int, addr: int, length: int) -> bytes:
    return packet(motor_id, INST_READ, addr.to_bytes(2, "little") + length.to_bytes(2, "little"))


def write_packet(motor_id: int, addr: int, data: bytes) -> bytes:
    return packet(motor_id, INST_WRITE, addr.to_bytes(2, "little") + data)


def parse_status(raw: bytes, motor_id: int) -> bytes:
    """Return the parameters of a status packet from motor_id; raise ValueError on anything malformed."""
    raw = bytes(raw or b"")
    start = raw.find(b"\xff\xff\xfd\x00")
    if start < 0 or len(raw) < start + 11:
        raise ValueError("no reply")
    raw = raw[start:]
    length = int.from_bytes(raw[5:7], "little")
    end = 7 + length
    if len(raw) < end or raw[4] != motor_id or raw[7] != INST_STATUS:
        raise ValueError("unexpected reply")
    if crc16(raw[:end - 2]) != int.from_bytes(raw[end - 2:end], "little"):
        raise ValueError("bad checksum")
    if raw[8] & 0x7F:
        raise ValueError(f"motor reported error 0x{raw[8] & 0x7F:02x}")
    return raw[9:end - 2]


def decode(err_byte: int) -> list[str]:
    return [name for bit, name in ERR_BITS.items() if err_byte & (1 << bit)]


class MotorHealth:
    def __init__(self, url: str = RAW_URL, connect=None):
        self.url = url
        self._connect = connect          # tests inject a fake; default opens the daemon websocket

    def _open(self):
        if self._connect:
            return self._connect()
        from websockets.sync.client import connect
        return connect(self.url, open_timeout=5, max_size=None)

    @staticmethod
    def _xfer(ws, pkt: bytes, motor_id: int) -> bytes:
        ws.send(pkt)
        return parse_status(ws.recv(timeout=2), motor_id)

    def _errors(self, ws, motor_id: int) -> list[str]:
        errs = decode(self._xfer(ws, read_packet(motor_id, ADDR_HW_ERROR, 1), motor_id)[0])
        if "Input voltage" in errs:
            try:
                volts = int.from_bytes(self._xfer(ws, read_packet(motor_id, ADDR_VOLTAGE, 2), motor_id), "little") / 10
                if volts <= MAX_VOLTAGE:
                    errs.remove("Input voltage")
            except ValueError:
                pass
        return errs

    def read(self) -> dict:
        """{"motors": {name: {"label", "id", "errors": [...]} or {"error": "no reply"}}, "faults": [names]}"""
        motors, faults = {}, []
        with self._open() as ws:
            for name, mid in MOTORS.items():
                entry = {"label": LABELS[name], "id": mid}
                try:
                    entry["errors"] = self._errors(ws, mid)
                    if entry["errors"]:
                        faults.append(name)
                except Exception as e:
                    entry["errors"] = []
                    entry["unreadable"] = str(e)[:80]
                motors[name] = entry
        return {"motors": motors, "faults": faults}

    def clear(self, names: list[str] | None = None) -> dict:
        """Reboot the motors that report an error (or the named ones) and restore their PID gains.

        Torque is off after a reboot: call the robot's enable_motors() afterwards to hold them in place again."""
        if names is None:
            names = self.read()["faults"]
        names = [n for n in names if n in MOTORS]
        results = {}
        if not names:
            return {"cleared": results, "faults": []}
        with self._open() as ws:
            for name in names:
                mid = MOTORS[name]
                res = {"label": LABELS[name]}
                try:
                    res["before"] = self._errors(ws, mid)
                    pid = self._xfer(ws, read_packet(mid, ADDR_PID, 6), mid)
                    ws.send(packet(mid, INST_REBOOT))
                    try:
                        ws.recv(timeout=2)       # the reboot acknowledgement (may carry the old alert bit)
                    except Exception:
                        pass
                    time.sleep(REBOOT_WAIT_S)
                    if len(pid) == 6:
                        self._xfer(ws, write_packet(mid, ADDR_PID, pid), mid)
                    res["after"] = self._errors(ws, mid)
                    res["ok"] = not res["after"]
                    log.info("motor %s (id %d) rebooted: %s -> %s", name, mid, res["before"], res["after"])
                except Exception as e:
                    res["ok"] = False
                    res["error"] = str(e)[:120]
                    log.warning("could not clear motor %s (id %d): %s", name, mid, e)
                results[name] = res
        return {"cleared": results, "faults": [n for n, r in results.items() if not r.get("ok")]}
