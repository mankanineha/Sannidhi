#!/usr/bin/env python3
"""Inspect FallGuardNet outputs for complete, non-overlapping live CSI windows.

Run the dashboard backend first, then run this script and perform the named
ordinary activity while exactly one 950-frame window is collected.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

API_BASE = "http://127.0.0.1:8000"
WINDOW_SIZE = 950
FEATURE_COUNT = 52
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = {}
MODEL_RESULTS = {}


def api_get(path, timeout=5):
    try:
        with urllib.request.urlopen(f"{API_BASE}{path}", timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as error:
        return {"_error": str(error)}


def grab_live_matrix():
    result = api_get("/api/capture", timeout=10)
    if "_error" in result or "error" in result:
        return None, result.get("_error") or result.get("error")
    rows = np.asarray(result.get("rows", []), dtype=np.float32)
    if rows.shape != (WINDOW_SIZE, FEATURE_COUNT):
        return None, f"unexpected capture shape {rows.shape}"
    return rows, None


def wait_for_next_window(label):
    """Wait by model-window index, not by elapsed seconds."""
    before = api_get("/api/pipeline_info", timeout=2)
    if "_error" in before or "error" in before:
        print(f"[ERROR] Cannot read backend window state: {before}")
        return None, None
    if before.get("window_mode") != "non-overlapping, fixed-size windows":
        print("[ERROR] Backend is not running the expected non-overlapping window version.")
        print("        Stop the old backend process and launch dashboard/backend/app.py again.")
        return None, None

    target = int(before.get("last_completed_window_index", 0)) + 1
    print(f"\nCollecting '{label}' in model window #{target} ({WINDOW_SIZE} frames).")
    step = 0
    while True:
        status = api_get("/api/pipeline_info", timeout=2)
        completed = int(status.get("last_completed_window_index", target - 1))
        buffered = int(status.get("active_buffer_size", 0))
        print(f"\r  Window {completed}/{target} · next input {buffered}/{WINDOW_SIZE} frames", end="", flush=True)
        if completed >= target:
            break
        step += 1
        time.sleep(0.2)  # Polling only; the capture boundary is the frame count.
    print()

    capture, error = grab_live_matrix()
    model = api_get("/api/live_model", timeout=8)
    if capture is None or "_error" in model or "error" in model:
        print(f"[ERROR] capture/model result unavailable: {error or model}")
        return None, None
    if model.get("window_index") != target:
        print(f"[ERROR] Requested window #{target}, backend returned #{model.get('window_index')}.")
        return None, None
    return capture, model


def matrix_stats(matrix):
    zeros_by_feature = np.mean(matrix == 0, axis=0)
    return {
        "shape": list(matrix.shape),
        "mean": round(float(np.mean(matrix)), 4),
        "std": round(float(np.std(matrix)), 4),
        "min": round(float(np.min(matrix)), 4),
        "max": round(float(np.max(matrix)), 4),
        "zero_fraction": round(float(np.mean(matrix == 0)), 6),
        "zero_features": [int(i) for i, value in enumerate(zeros_by_feature) if value > 0.5],
    }


def report_window(label, matrix, model):
    stats = matrix_stats(matrix)
    RECORDINGS[label] = matrix
    MODEL_RESULTS[label] = model
    print(f"\n{label}: window #{model['window_index']} · frames {model['first_frame']}-{model['last_frame']}")
    print(f"  FallGuardNet v2: {model['prediction']} · fall probability {100 * model['probability_fall']:.4f}%")
    print(f"  logits: non-fall={model['logit_nonfall']:.6f}, fall={model['logit_fall']:.6f}")
    print(f"  CSI input: {stats['shape']} · zero fraction={stats['zero_fraction']:.6f}")
    print(f"  Channels zero for >50% of frames: {stats['zero_features'] or 'none'}")


def print_recent_window_history():
    result = api_get("/api/model_windows?limit=1000", timeout=8)
    if "_error" in result or "error" in result:
        print(f"[WARN] Could not read model-window history: {result}")
        return
    windows = result.get("windows", [])
    print(f"\nBackend history: {len(windows)} non-overlapping model windows")
    print(f"  Model FALL decisions: {result.get('fall_predictions', 0)}")
    for item in windows:
        print(
            f"  #{item['window_index']:>4} frames {item['first_frame']}-{item['last_frame']}: "
            f"{item['prediction']:<8} p_fall={item['probability_fall']:.8f} "
            f"zero={item.get('zero_fraction', 0):.6f}"
        )


def main():
    print("FallGuardNet v2 · fixed 950-frame, non-overlapping window audit")
    info = api_get("/api/pipeline_info")
    if "_error" in info or "error" in info:
        print(f"[ERROR] Cannot reach backend at {API_BASE}: {info}")
        print("Start the updated backend with: python3 dashboard/backend/app.py")
        sys.exit(1)
    if info.get("expected_shape") != [1, WINDOW_SIZE, FEATURE_COUNT]:
        print(f"[ERROR] Unexpected model input contract: {info.get('expected_shape')}")
        sys.exit(1)
    print(f"Backend: {info.get('model')} · input {info['expected_shape']} · {info.get('window_mode')}")

    reference = api_get("/api/test_mat", timeout=8)
    if "_error" not in reference and "error" not in reference:
        print(
            "Training-reference fall sample: "
            f"{reference.get('prediction')} · p_fall={reference.get('fall_probability')}"
        )

    for label, instruction in (
        ("sitting_still", "Sit still for this window"),
        ("walking", "Walk normally during this window"),
    ):
        input(f"\n{instruction}; press Enter to begin collecting the next window...")
        matrix, model = wait_for_next_window(label)
        if matrix is not None:
            report_window(label, matrix, model)

    print_recent_window_history()
    output = PROJECT_ROOT / "diagnostics" / "diagnostic_captures.json"
    saved = {
        label: {
            "model": MODEL_RESULTS[label],
            "stats": matrix_stats(matrix),
            "raw_matrix": matrix.tolist(),
        }
        for label, matrix in RECORDINGS.items()
    }
    output.write_text(json.dumps(saved))
    print(f"\nSaved exact model windows to {output}")
    print("No motion heuristic in this report can produce a fall classification or alert.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[DIAG] Stopped.")
