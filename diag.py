import time, serial, sys, os
import numpy as np
import torch
from simulator import parse_csv_row, FEAT, WIN
from detector import load_models, preprocess

fallnet, fn_thresh, spectranet, sn_thresh, mu, sd, sn_mu, sn_sd = load_models()

ser = serial.Serial(os.getenv("ESP32_PORT", "/dev/cu.usbserial-0001"), 921600, timeout=1)
time.sleep(0.5)
ser.flushInput()

window = None
frame_idx = 0
start = time.time()

fn_history = []
sn_history = []
surge_history = []

_ACTIVE_THRESHOLD = 2.0
_MIN_ACTIVE = 80

def compute_kinematics_debug(window):
    amp = window[:, :116]
    active_counts = np.array([(row > _ACTIVE_THRESHOLD).sum() for row in amp])
    full_mask = active_counts >= _MIN_ACTIVE
    pkt_mean = np.mean(amp, axis=1, keepdims=True) + 1e-6
    norm_amp = amp / pkt_mean
    valid_diffs = []
    mixed_diffs = []
    for i in range(1, len(norm_amp)):
        d = float(np.mean(np.abs(norm_amp[i] - norm_amp[i-1])))
        if full_mask[i] == full_mask[i-1]:
            valid_diffs.append(d)
        else:
            mixed_diffs.append(d)
    return valid_diffs, mixed_diffs

print("=== COMPREHENSIVE DIAGNOSTIC: 25 seconds, quiet room ===")
print("DO NOT MOVE. Just sit or stand still in the room.")
print("FN_thresh=%.2f  SN_thresh=%.2f" % (fn_thresh, sn_thresh))
print()
print("%6s %6s %6s | %8s %8s | %10s | %s" % ("Frame","FN","SN","energy","surge","mix_surge","decision"))
print("-" * 80)

while time.time() - start < 25:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if 'CSI_TENSOR' in line:
        row = line[line.find('CSI_TENSOR'):].split(',')
        arr = parse_csv_row(row)
        if arr is not None:
            frame_idx += 1
            if window is None:
                window = np.tile(arr, (WIN, 1))
            else:
                window = np.roll(window, -1, axis=0)
                window[-1] = arr

            if frame_idx < 50:
                continue

            if frame_idx % 8 == 0:
                tensor = preprocess(window, mu, sd)
                with torch.no_grad():
                    fn_prob = torch.sigmoid(fallnet(tensor)[0, 0]).item()
                    sn_tensor = preprocess(window, sn_mu, sn_sd)
                    sn_prob = torch.softmax(spectranet(sn_tensor), dim=1)[0, 1].item()

                valid_diffs, mixed_diffs = compute_kinematics_debug(window)
                recent_valid = valid_diffs[-100:] if valid_diffs else [0]
                energy = float(np.mean(recent_valid))
                surge  = float(np.max(recent_valid)) if recent_valid else 0.0
                mixed_surge = float(np.max(mixed_diffs[-100:])) if mixed_diffs else 0.0

                fn_history.append(fn_prob)
                sn_history.append(sn_prob)
                surge_history.append(surge)

                model_ok = (fn_prob >= fn_thresh) and (sn_prob >= sn_thresh)
                gate_ok  = (energy >= 0.10) and (surge >= 0.45)
                decision = "WOULD FIRE" if (model_ok and gate_ok) else "SAFE"

                print("%6d %6.3f %6.3f | %8.3f %8.3f | %10.3f | %s" % (
                    frame_idx, fn_prob, sn_prob, energy, surge, mixed_surge, decision))

print()
print("=== NOISE FLOOR SUMMARY ===")
if fn_history:
    print("FallNet:    min=%.3f  max=%.3f  mean=%.3f  std=%.3f" % (min(fn_history), max(fn_history), np.mean(fn_history), np.std(fn_history)))
    print("SpectraNet: min=%.3f  max=%.3f  mean=%.3f  std=%.3f" % (min(sn_history), max(sn_history), np.mean(sn_history), np.std(sn_history)))
    print("Surge:      min=%.3f  max=%.3f  mean=%.3f  std=%.3f" % (min(surge_history), max(surge_history), np.mean(surge_history), np.std(surge_history)))
    print()
    print(">>> Recommended FallNet threshold: > %.3f" % (max(fn_history) + 0.08))
    print(">>> Recommended Surge   threshold: > %.3f" % (max(surge_history) + 0.08))
else:
    print("NO CSI_TENSOR rows were received. This is a serial-connection issue, not a model result.")
    print("Close Arduino Serial Monitor, unplug/replug the ESP32, then run test_serial.py.")

ser.close()
