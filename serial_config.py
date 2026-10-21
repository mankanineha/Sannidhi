"""Serial-port selection shared by the dashboard and CLI."""

from __future__ import annotations

import os

from serial.tools import list_ports


def available_ports():
    return list(list_ports.comports())


def resolve_serial_port() -> str | None:
    """Return an explicit port or the best USB serial candidate.

    Set SANNIDHI_SERIAL_PORT to a value such as COM7 to force a port. With
    AUTO, Bluetooth virtual ports are ignored so they are not mistaken for
    an ESP32 connection.
    """
    configured = os.getenv("SANNIDHI_SERIAL_PORT", "AUTO").strip()
    if configured and configured.upper() != "AUTO":
        return configured

    candidates = []
    for port in available_ports():
        text = f"{port.device} {port.description} {port.hwid}".lower()
        if "bluetooth" in text or "bthenum" in text:
            continue
        candidates.append(port.device)

    return candidates[0] if len(candidates) == 1 else None


def describe_ports() -> str:
    ports = available_ports()
    if not ports:
        return "none"
    return "; ".join(f"{p.device} ({p.description})" for p in ports)
