import socket
import numpy as np
import onnxruntime as ort
import os
import time
import sys
from serial_config import describe_ports, resolve_serial_port

# --- CONFIGURATION ---
SERIAL_PORT = resolve_serial_port()
BAUD_RATE = int(os.getenv("SANNIDHI_BAUD_RATE", "921600"))

# FallGuardNet v2 has a fixed input contract of 950 CSI frames x 52 features.
WINDOW_SIZE = 950   # Must match training: 950 frames × 52 subcarriers
FEATURE_COUNT = 52  # ESP-Fi-HAR amplitude subcarriers (no dead-feature mask)
N_SUBCARRIERS = 52

# FallGuardNet v2 weights (distilled from ResNet18 teacher)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "esp32_fallguardnet_v2.onnx")

# --- LOAD MODEL ---
print(f"Loading FallGuardNet v2 ONNX model from:\n  {MODEL_PATH}\n")
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
    expected_shape = ort_session.get_inputs()[0].shape
    if len(expected_shape) != 3 or expected_shape[1:] != [WINDOW_SIZE, FEATURE_COUNT]:
        raise ValueError(f"Expected ONNX input [batch, {WINDOW_SIZE}, {FEATURE_COUNT}], got {expected_shape}")
    print(f"✅ Model loaded. Input: '{input_name}' shape={expected_shape}\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# --- Multilingual Voice Triage & Cloud Caregiver App Integration ---
sys.path.append(os.path.join(os.path.dirname(__file__), "dashboard", "backend"))
try:
    from voice_agent import voice_agent
    HAS_VOICE_AGENT = True
    print("[VoiceAgent] ✅ Multilingual Voice Triage Engine loaded.")
except Exception as e:
    voice_agent = None
    HAS_VOICE_AGENT = False
    print(f"[VoiceAgent] ⚠️  VoiceAgent load failed: {e}")

try:
    import firebase_client
    HAS_FIREBASE = True
    firebase_client.init_firebase()
    print("[Firebase] ✅ Cloud Firestore Caregiver Sync initialized.")
except Exception as e:
    firebase_client = None
    HAS_FIREBASE = False
    print(f"[Firebase] ⚠️  Firebase init failed: {e}")

# --- CONNECT TO HARDWARE ---
if SERIAL_PORT is None:
    print("Hardware connection failed: no USB serial ESP32 port was detected.")
    print(f"Detected ports: {describe_ports()}")
    print("Set SANNIDHI_SERIAL_PORT=COMx after checking Device Manager.")
    sys.exit(2)

print(f"Connecting to ESP32 on {SERIAL_PORT} at {BAUD_RATE} baud...")
try:
    import serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.flushInput()
    print("Successfully connected to hardware!\n")
except Exception as e:
    print(f"Hardware connection failed: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Motion scoring — mirrors app.py for behavioural consistency
# ---------------------------------------------------------------------------

SHORT_WINDOW = 20  # Display-only motion context over 20 received frames.

def _get_clean_tail(raw_matrix: np.ndarray, window: int) -> np.ndarray:
    """Extract recent frames with 3-frame median filter to kill Wi-Fi jitter."""
    tail = raw_matrix[-window:, :].copy()
    if len(tail) >= 5:
        padded = np.pad(tail, ((1, 1), (0, 0)), mode="edge")
        tail = np.median([padded[:-2], padded[1:-1], padded[2:]], axis=0)
    return tail


def compute_motion_score(raw_matrix: np.ndarray) -> float:
    """0.4 s RMS motion — primary state-machine signal."""
    tail = _get_clean_tail(raw_matrix, 40)
    frame_means = np.mean(tail, axis=1, keepdims=True) + 1e-6
    tail_norm = tail / frame_means
    diffs = np.abs(np.diff(tail_norm, axis=0))
    rms = np.sqrt(np.mean(diffs ** 2, axis=1))
    return float(np.percentile(rms, 85) * 2.5)


def compute_motion_short(raw_matrix: np.ndarray) -> float:
    """Display-only motion context over the last 20 received frames."""
    tail = _get_clean_tail(raw_matrix, SHORT_WINDOW)
    frame_means = np.mean(tail, axis=1, keepdims=True) + 1e-6
    tail_norm = tail / frame_means
    per_sc_std = np.std(tail_norm, axis=0)
    return float(np.sqrt(np.mean(per_sc_std ** 2)) * 2.5)


def compute_motion_instant(raw_matrix: np.ndarray) -> float:
    """Frame-to-frame velocity from 3-frame medians."""
    if len(raw_matrix) < 6:
        return 0.0
    curr = np.median(raw_matrix[-3:, :], axis=0)
    prev = np.median(raw_matrix[-6:-3, :], axis=0)
    curr_norm = curr / (np.mean(curr) + 1e-6)
    prev_norm = prev / (np.mean(prev) + 1e-6)
    return float(np.sqrt(np.mean((np.abs(curr_norm - prev_norm)) ** 2)) * 2.5)


def compute_energy_delta(raw_matrix: np.ndarray) -> float:
    """Per-subcarrier relative RMS change over 0.2 s."""
    tail = _get_clean_tail(raw_matrix, SHORT_WINDOW)
    frame_means = np.mean(tail, axis=1, keepdims=True) + 1e-6
    tail_norm = tail / frame_means
    per_sc_std = np.std(tail_norm, axis=0)
    return float(np.sqrt(np.mean(per_sc_std ** 2)) * 10.0)


# ---------------------------------------------------------------------------
# Normalisation — MUST match ESP-Fi-HAR training: single global Z-score
# per window, not per-feature.
# ---------------------------------------------------------------------------

def normalize_global(raw_matrix: np.ndarray) -> np.ndarray:
    """
    Global (window-level) Z-score normalisation.

    The FallGuardNet v2 model was trained with:
        x = (x - mean(x)) / (std(x) + 1e-8)
    applied to the entire (950, 52) window as one flat distribution.
    Per-feature normalisation would shift the input distribution and
    cause the model to produce unstable / incorrect probabilities.
    """
    mean = np.mean(raw_matrix)
    std  = np.std(raw_matrix) + 1e-8
    normalized = (raw_matrix - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Inference loop: one model decision per non-overlapping 950-frame window
csi_buffer: list[list[float]] = []
window_index = 0
WINDOW_STRIDE = int(os.getenv("WINDOW_STRIDE", "50"))
frames_since_infer = 0
fall_alert_active = False
print(f"Collecting sliding model windows of {WINDOW_SIZE} frames (stride={WINDOW_STRIDE})...\n")

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if "CSI_TENSOR" not in line:
            continue

        try:
            parts = line.split(",")
            if len(parts) < FEATURE_COUNT + 3:
                continue
            features = [float(value) for value in parts[3:FEATURE_COUNT + 3]]
            if len(features) != FEATURE_COUNT:
                continue

            csi_buffer.append(features)
            if len(csi_buffer) > WINDOW_SIZE:
                csi_buffer.pop(0)

            if len(csi_buffer) < WINDOW_SIZE:
                if len(csi_buffer) % 50 == 0:
                    print(f"Collecting model window: {len(csi_buffer)}/{WINDOW_SIZE} frames", end="\r")
                continue

            frames_since_infer += 1
            if frames_since_infer < WINDOW_STRIDE and window_index > 0:
                continue
            frames_since_infer = 0

            # Freeze this block and run inference on the latest 950 frames
            raw_matrix = np.nan_to_num(np.asarray(csi_buffer, dtype=np.float32), nan=0.0)
            window_index += 1

            normalized = normalize_global(raw_matrix)
            input_tensor = normalized[np.newaxis, :, :]
            logits = np.asarray(ort_session.run(None, {input_name: input_tensor})[0][0], dtype=np.float64)
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            fall_probability = float(probabilities[1])
            prediction = "FALL" if logits[1] > logits[0] else "NON-FALL"

            # Motion remains diagnostic context only. It cannot produce a fall decision.
            motion_short = compute_motion_short(raw_matrix)
            energy_delta = compute_energy_delta(raw_matrix)
            timestamp = time.strftime("%H:%M:%S")
            print(
                f"\n[{timestamp}] Window #{window_index} ({WINDOW_SIZE} frames): "
                f"{prediction} · FallGuardNet p={fall_probability*100:.2f}% · "
                f"motion={motion_short:.3f}"
            )

            if prediction == "FALL":
                if HAS_VOICE_AGENT and voice_agent and not voice_agent.is_active and not fall_alert_active:
                    voice_agent.trigger_confirmation_flow(metrics={
                        "probability": fall_probability,
                        "fallnet_prob": fall_probability,
                        "spectranet_prob": None,
                        "motion_energy": energy_delta,
                        "energy_delta": energy_delta,
                        "confirmation_source": "FallGuardNet v2 single-window inference",
                        "model_window_index": window_index,
                    })
                fall_alert_active = True
            else:
                fall_alert_active = False

        except (ValueError, IndexError) as error:
            print(f"\nSkipping malformed CSI record: {error}")

except KeyboardInterrupt:
    print("\n\nShutting down gracefully…")
    ser.close()
    sys.exit(0)
