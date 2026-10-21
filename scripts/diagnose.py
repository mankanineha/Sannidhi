"""Capture reproducible startup diagnostics for the Sannidhi dashboard or CLI.

This wrapper deliberately does not modify the application. It records startup
checks, child-process output, and periodic heartbeats so a freeze can be
reported with evidence instead of only a screenshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]


def stamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_event(log, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def package_version(name: str):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # diagnostic output must never stop the run
        return f"unavailable: {type(exc).__name__}: {exc}"


def collect_preflight(port: str, baud: int, mode: str) -> dict:
    result = {
        "timestamp": stamp(),
        "mode": mode,
        "project": str(PROJECT),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "port": port,
        "baud": baud,
        "files": {},
        "packages": {},
        "serial_ports": [],
    }

    for relative in (
        "cli_inference.py",
        "dashboard/backend/app.py",
        "csi_spectranet_best.onnx",
        "dashboard/frontend/index.html",
    ):
        path = PROJECT / relative
        result["files"][relative] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
        }

    for name in ("numpy", "serial", "fastapi", "uvicorn", "onnxruntime"):
        result["packages"][name] = package_version(name)

    try:
        from serial.tools import list_ports

        result["serial_ports"] = [
            {
                "device": item.device,
                "description": item.description,
                "hwid": item.hwid,
            }
            for item in list_ports.comports()
        ]
    except Exception as exc:
        result["serial_ports_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import serial

        started = time.monotonic()
        connection = serial.Serial(port, baudrate=baud, timeout=0.2)
        connection.close()
        result["serial_probe"] = {
            "open": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        result["serial_probe"] = {
            "open": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return result


def command_for(mode: str, port: str, baud: int):
    if mode == "dashboard":
        cwd = PROJECT / "dashboard" / "backend"
        command = [sys.executable, "-u", "app.py"]
    else:
        cwd = PROJECT
        command = [sys.executable, "-u", "cli_inference.py"]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SANNIDHI_DIAGNOSTICS"] = "1"
    env["SANNIDHI_SERIAL_PORT"] = port
    env["SANNIDHI_BAUD_RATE"] = str(baud)
    return command, cwd, env


def stream_output(pipe, stream_name: str, log, stop_event: threading.Event):
    try:
        for line in iter(pipe.readline, ""):
            write_event(log, f"{stream_name}: {line.rstrip()}" )
    except Exception:
        write_event(log, f"{stream_name} reader error:\n{traceback.format_exc().rstrip()}")
    finally:
        stop_event.set()


def process_snapshot(process: subprocess.Popen) -> str:
    if process.poll() is not None:
        return f"exited rc={process.returncode}"
    try:
        tasklist = subprocess.run(
            ["tasklist", "/FI", f"PID eq {process.pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        detail = tasklist.stdout.strip().replace("\n", " | ")
        return f"running pid={process.pid} {detail}".strip()
    except Exception as exc:
        return f"running pid={process.pid}; tasklist unavailable: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dashboard", "cli"))
    parser.add_argument("--port", default=os.getenv("SANNIDHI_SERIAL_PORT", "COM5"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("SANNIDHI_BAUD_RATE", "921600")))
    parser.add_argument("--duration", type=int, default=60, help="seconds to observe; Ctrl+C also stops cleanly")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "diagnostics")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    preflight_path = run_dir / "preflight.json"

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        write_event(log, f"Starting diagnostics mode={args.mode} duration={args.duration}s")
        write_event(log, f"Run directory: {run_dir}")

        preflight = collect_preflight(args.port, args.baud, args.mode)
        preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        write_event(log, f"Preflight written: {preflight_path}")
        write_event(log, f"Serial probe: {preflight.get('serial_probe')}")
        write_event(log, f"Visible serial ports: {preflight.get('serial_ports')}")
        write_event(log, f"Launching with cwd={command_for(args.mode, args.port, args.baud)[1]}")

        command, cwd, env = command_for(args.mode, args.port, args.baud)
        process = None
        stop_event = threading.Event()
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            write_event(log, f"Child started pid={process.pid} command={command}")
            threads = [
                threading.Thread(target=stream_output, args=(process.stdout, "stdout", log, stop_event), daemon=True),
                threading.Thread(target=stream_output, args=(process.stderr, "stderr", log, stop_event), daemon=True),
            ]
            for thread in threads:
                thread.start()

            while True:
                if process.poll() is not None:
                    write_event(log, f"Child finished rc={process.returncode}")
                    break
                elapsed = time.monotonic() - started
                write_event(log, f"HEARTBEAT elapsed={elapsed:.1f}s {process_snapshot(process)}")
                if elapsed >= args.duration:
                    write_event(log, "Observation duration reached; terminating child")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        write_event(log, "Child did not terminate; killing it")
                        process.kill()
                        process.wait(timeout=5)
                    write_event(log, f"Child stopped rc={process.returncode}")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            write_event(log, "Ctrl+C received; stopping child")
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            write_event(log, f"Wrapper error:\n{traceback.format_exc().rstrip()}")
            if process and process.poll() is None:
                process.kill()
            return 2

    print(f"Diagnostics saved to: {run_dir}")
    print(f"Attach these files to the issue: {log_path.name}, {preflight_path.name}")
    return process.returncode if process is not None and process.returncode is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
