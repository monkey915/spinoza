"""Low-level Feetech STS3215 serial bus communication.

Requires the ``scservo_sdk`` package (pip install scservo_sdk).
"""

from __future__ import annotations

import time
from typing import Optional

try:
    import scservo_sdk as scs
except ImportError:
    scs = None  # allow importing for tests / offline development

from . import config


# STS3215 control table addresses
_ADDR_TORQUE_ENABLE = 40
_ADDR_GOAL_POSITION = 42
_ADDR_GOAL_SPEED    = 46
_ADDR_PRESENT_POSITION = 56
_ADDR_PRESENT_SPEED    = 58
_ADDR_PRESENT_LOAD     = 60
_ADDR_TORQUE_LIMIT     = 48


class FeetechBus:
    """Communicate with STS3215 servos on a serial TTL bus.

    Usage::

        bus = FeetechBus()
        bus.open()
        bus.ping(1)
        bus.set_torque(1, True)
        bus.write_position(1, 2048)
        print(bus.read_position(1))
        bus.close()
    """

    def __init__(self, port: str | None = None, baudrate: int | None = None):
        if scs is None:
            raise RuntimeError(
                "scservo_sdk not installed. Run: pip install scservo_sdk"
            )
        self.port = port or config.SERIAL_PORT
        self.baudrate = baudrate or config.BAUDRATE
        self._port_handler: Optional[scs.PortHandler] = None
        self._packet_handler: Optional[scs.PacketHandler] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def open(self):
        """Open the serial port and initialize the packet handler."""
        self._port_handler = scs.PortHandler(self.port)
        self._packet_handler = scs.PacketHandler(0)  # protocol version 0 for STS

        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port}")
        if not self._port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baud rate {self.baudrate}")
        print(f"Servo bus opened: {self.port} @ {self.baudrate}")

    def close(self):
        """Close the serial port."""
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
        print("Servo bus closed")

    # ------------------------------------------------------------------
    # Low-level read / write
    # ------------------------------------------------------------------

    def _write2(self, servo_id: int, addr: int, value: int):
        result, error = self._packet_handler.write2ByteTxRx(
            self._port_handler, servo_id, addr, value
        )
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"Write failed [servo {servo_id}, addr {addr}]: "
                f"{self._packet_handler.getTxRxResult(result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"Servo error [servo {servo_id}]: "
                f"{self._packet_handler.getRxPacketError(error)}"
            )

    def _read2(self, servo_id: int, addr: int) -> int:
        value, result, error = self._packet_handler.read2ByteTxRx(
            self._port_handler, servo_id, addr
        )
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"Read failed [servo {servo_id}, addr {addr}]: "
                f"{self._packet_handler.getTxRxResult(result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"Servo error [servo {servo_id}]: "
                f"{self._packet_handler.getRxPacketError(error)}"
            )
        return value

    def _write1(self, servo_id: int, addr: int, value: int):
        result, error = self._packet_handler.write1ByteTxRx(
            self._port_handler, servo_id, addr, value
        )
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"Write failed [servo {servo_id}, addr {addr}]: "
                f"{self._packet_handler.getTxRxResult(result)}"
            )

    # ------------------------------------------------------------------
    # Servo operations
    # ------------------------------------------------------------------

    def ping(self, servo_id: int) -> bool:
        """Ping a servo. Returns True if it responds."""
        _, result, _ = self._packet_handler.ping(
            self._port_handler, servo_id
        )
        return result == scs.COMM_SUCCESS

    def set_torque(self, servo_id: int, enable: bool):
        """Enable or disable torque on a servo."""
        self._write1(servo_id, _ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def set_torque_limit(self, servo_id: int, limit: int | None = None):
        """Set the torque limit (0-1000)."""
        limit = limit if limit is not None else config.TORQUE_LIMIT
        self._write2(servo_id, _ADDR_TORQUE_LIMIT, min(max(limit, 0), 1000))

    def write_position(self, servo_id: int, position: int,
                       speed: int | None = None):
        """Command a servo to a raw position (0-4095).

        Optionally set the movement speed (0-4095).
        """
        position = min(max(position, 0), 4095)
        if speed is not None:
            self._write2(servo_id, _ADDR_GOAL_SPEED, min(max(speed, 0), 4095))
        self._write2(servo_id, _ADDR_GOAL_POSITION, position)

    def read_position(self, servo_id: int) -> int:
        """Read the current servo position (0-4095)."""
        return self._read2(servo_id, _ADDR_PRESENT_POSITION)

    def read_speed(self, servo_id: int) -> int:
        """Read the current servo speed."""
        return self._read2(servo_id, _ADDR_PRESENT_SPEED)

    def read_load(self, servo_id: int) -> int:
        """Read the current servo load."""
        return self._read2(servo_id, _ADDR_PRESENT_LOAD)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def ping_all(self) -> dict[str, bool]:
        """Ping all configured servos. Returns {name: reachable}."""
        results = {}
        for name, sid in config.SERVO_IDS.items():
            results[name] = self.ping(sid)
        return results

    def torque_all(self, enable: bool):
        """Enable or disable torque on all servos."""
        for name, sid in config.SERVO_IDS.items():
            self.set_torque(sid, enable)

    def read_all_positions(self) -> dict[str, int]:
        """Read raw positions of all servos."""
        positions = {}
        for name, sid in config.SERVO_IDS.items():
            positions[name] = self.read_position(sid)
        return positions

    def emergency_stop(self):
        """Disable torque on all servos immediately."""
        for name, sid in config.SERVO_IDS.items():
            try:
                self.set_torque(sid, False)
            except RuntimeError:
                pass  # best effort
        print("EMERGENCY STOP: all servos torque disabled")
