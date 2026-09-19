"""
ESP32 CSI Streamer for Sannidhi Fall Detection Pipeline
=========================================================
Supports 3 modes:
  --mode replay   : Stream raw_capture.csv at real 100 Hz rate (default)
  --mode synthetic: Generate realistic synthetic CSI, press F to inject a fall
  --mode hardware : Connect directly to physical ESP32 via USB serial port
"""

from __future__ import annotations
import csv
import time
import threading
import numpy as np
import queue
import sys
import os
import glob
from typing import Optional

WIN      = 500
FEAT     = 232
N_SUB    = 116
HZ       = 100
DELAY    = 1.0 / HZ

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "sannidhi-main", "raw_capture.csv")

csi_queue: queue.Queue = queue.Queue(maxsize=2000)
_stop_event = threading.Event()
_fall_inject = threading.Event()


def find_esp32_port() -> Optional[str]:
    patterns = [
        "/dev/cu.usbserial*",
        "/dev/cu.wchusbserial*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*"
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def parse_csv_row(row: list) -> Optional[np.ndarray]:
    if not row:
        return None
    try:
        if row[0].strip() == "CSI_TENSOR":
            raw_vals = row[3:]
        else:
            raw_vals = row

        values = []
        for v in raw_vals:
            try:
                values.append(float(v.strip()))
            except ValueError:
                continue

        if len(values) < 10:
            return None

        if len(values) < FEAT:
            values = values + [0.0] * (FEAT - len(values))
        elif len(values) > FEAT:
            values = values[:FEAT]

        arr = np.array(values, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        return None


def run_replay(verbose=True):
    if not os.path.exists(CSV_PATH):
        print(f"[Simulator] ❌ CSV not found: {CSV_PATH}")
        _stop_event.set()
        return

    print(f"[Simulator] 📡 Replay mode — streaming {CSV_PATH} at {HZ} Hz")
    frame_count, loop_count = 0, 0

    while not _stop_event.is_set():
        loop_count += 1
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if _stop_event.is_set():
                    return
                arr = parse_csv_row(row)
                if arr is None:
                    continue

                t0 = time.perf_counter()
                try:
                    csi_queue.put_nowait(arr)
                except queue.Full:
                    try:
                        csi_queue.get_nowait()
                        csi_queue.put_nowait(arr)
                    except queue.Empty:
                        pass

                frame_count += 1
                if verbose and frame_count % 500 == 0:
                    print(f"[Simulator] ✅ Sent {frame_count} frames (loop {loop_count}), queue={csi_queue.qsize()}")

                elapsed = time.perf_counter() - t0
                sleep_t = max(0.0, DELAY - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)


def _make_ambient_frame(t: float) -> np.ndarray:
    rng = np.random.default_rng(int(t * 1000) % 2**31)
    amp = np.zeros(N_SUB, dtype=np.float32)
    for i in range(N_SUB):
        amp[i] = 25.0 + 30.0 * (i / N_SUB) + rng.normal(0, 0.5)
    phase = np.zeros(N_SUB, dtype=np.float32)
    for i in range(N_SUB):
        phase[i] = -2.0 + 4.0 * (i / N_SUB) + rng.normal(0, 0.05)
    return np.concatenate([amp, phase]).astype(np.float32)


def _make_fall_frame(t: float, progress: float) -> np.ndarray:
    base = _make_ambient_frame(t)
    amp, phase = base[:N_SUB].copy(), base[N_SUB:].copy()
    if progress < 0.3:
        scale = progress / 0.3
        amp += scale * np.random.normal(15, 5, N_SUB).astype(np.float32)
        phase += scale * np.random.normal(0, 0.8, N_SUB).astype(np.float32)
    elif progress < 0.6:
        amp += np.random.normal(20, 8, N_SUB).astype(np.float32)
        phase += np.random.normal(0, 1.2, N_SUB).astype(np.float32)
    else:
        scale = 1.0 - (progress - 0.6) / 0.4
        amp += scale * np.random.normal(10, 3, N_SUB).astype(np.float32)
        phase += scale * np.random.normal(0, 0.4, N_SUB).astype(np.float32)
    return np.concatenate([amp, phase]).astype(np.float32)


def run_synthetic(verbose=True):
    print("[Simulator] 🤖 Synthetic mode — generating realistic CSI")
    frame_count = 0
    fall_duration = int(2.5 * HZ)
    fall_remaining = 0

    while not _stop_event.is_set():
        t = time.perf_counter()
        if _fall_inject.is_set():
            _fall_inject.clear()
            fall_remaining = fall_duration
            print("\n[Simulator] 🚨 FALL INJECTED — streaming fall signature for 2.5s")

        if fall_remaining > 0:
            progress = 1.0 - (fall_remaining / fall_duration)
            arr = _make_fall_frame(t, progress)
            fall_remaining -= 1
        else:
            arr = _make_ambient_frame(t)

        try:
            csi_queue.put_nowait(arr)
        except queue.Full:
            try:
                csi_queue.get_nowait()
                csi_queue.put_nowait(arr)
            except queue.Empty:
                pass

        frame_count += 1
        if verbose and frame_count % 500 == 0:
            print(f"[Simulator] ✅ {frame_count} synthetic frames, queue={csi_queue.qsize()}")

        elapsed = time.perf_counter() - t
        time.sleep(max(0.0, DELAY - elapsed))


def run_hardware(verbose: bool = True, port: str = None, baud: int = 921600):
    try:
        import serial
    except ImportError:
        print("[Simulator] ❌ pyserial not installed. Run: pip install pyserial")
        _stop_event.set()
        return

    if not port:
        port = find_esp32_port()

    if not port:
        print("[Simulator] ❌ No connected ESP32 found! Check USB cable and run 'ls /dev/cu.*'")
        _stop_event.set()
        return

    # Match arduino_csi_radar.ino. The host and device must use the same rate;
    # a lower rate cannot carry complete 232-feature text rows reliably.
    print(f"[Simulator] 🔌 Hardware mode — connecting to ESP32 on {port} @ {baud} baud")

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(1.5)
        ser.flushInput()
        print(f"[Simulator] ✅ Connected to ESP32 hardware on {port}!")
    except Exception as e:
        print(f"[Simulator] ❌ Failed to open serial port {port}: {e}")
        _stop_event.set()
        return

    frame_count = 0
    fall_duration = int(2.5 * HZ)
    fall_remaining = 0
    buffer = bytearray()

    while not _stop_event.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buffer.extend(chunk)

            while b'\n' in buffer:
                line_bytes, buffer = buffer.split(b'\n', 1)
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                if "CSI_TENSOR" in line:
                    idx = line.find("CSI_TENSOR")
                    row = line[idx:].split(",")
                else:
                    row = line.split(",")
                arr = parse_csv_row(row)
                if arr is not None:
                    if _fall_inject.is_set():
                        _fall_inject.clear()
                        fall_remaining = fall_duration
                        print("\n[Simulator] 🚨 FALL INJECTED over live hardware stream!")

                    if fall_remaining > 0:
                        progress = 1.0 - (fall_remaining / fall_duration)
                        fall_sig = _make_fall_frame(time.perf_counter(), progress)
                        arr = (0.3 * arr + 0.7 * fall_sig).astype(np.float32)
                        fall_remaining -= 1

                    try:
                        csi_queue.put_nowait(arr)
                    except queue.Full:
                        try:
                            csi_queue.get_nowait()
                            csi_queue.put_nowait(arr)
                        except queue.Empty:
                            pass
                    frame_count += 1
                    if verbose and (frame_count % 100 == 0 or frame_count <= 5):
                        print(f"[Simulator] 📡 Live ESP32: {frame_count} frames received | queue={csi_queue.qsize()}")
        except Exception as e:
            if "device reports readiness to read but returned no data" not in str(e):
                print(f"[Simulator] ⚠️ Serial read error: {e}")
            time.sleep(0.01)

    ser.close()
    print(f"[Simulator] Closed serial connection to {port}")


def trigger_fall():
    _fall_inject.set()

def stop():
    _stop_event.set()

def start(mode: str = "replay", verbose: bool = True) -> threading.Thread:
    _stop_event.clear()
    if mode == "hardware":
        fn = run_hardware
    elif mode == "synthetic":
        fn = run_synthetic
    else:
        fn = run_replay

    t = threading.Thread(target=fn, args=(verbose,), daemon=True, name=f"simulator-{mode}")
    t.start()
    return t
