"""
Final validation test: 
- 40 seconds of quiet room → must show ZERO false alarms
- Tests the full detector stack: FallNet threshold + adaptive gate
"""
import time, serial, sys, os
import numpy as np
import torch
from simulator import parse_csv_row, FEAT, WIN
from detector import load_models, preprocess, FALLNET_THRESH, SPECTRANET_THRESH, WARMUP_FRAMES
from physics_gate import PhysicsGate

fallnet, fn_thresh, spectranet, sn_thresh, mu, sd, sn_mu, sn_sd = load_models()

gate = PhysicsGate(
    min_motion_energy=0.05,
    min_impact_surge=0.40,
    zscore_threshold=3.5,
    baseline_frames=150,
    debounce_required=2,
    cooldown_sec=90.0
)

ser = serial.Serial(os.getenv("ESP32_PORT", "/dev/cu.usbserial-0001"), 921600, timeout=1)
time.sleep(0.5)
ser.flushInput()

window = None
frame_idx = 0
start = time.time()
false_alarms = 0
TEST_DURATION = 40

print("=== FINAL VALIDATION: %ds quiet-room, ZERO alarms expected ===" % TEST_DURATION)
print("FallNet threshold=%.2f | Z-score gate=3.5 sigma | Warmup=%d frames" % (fn_thresh, WARMUP_FRAMES))
print("DO NOT MOVE during this test.\n")

while time.time() - start < TEST_DURATION:
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

            if frame_idx < WARMUP_FRAMES:
                continue

            if frame_idx % 6 == 0:
                tensor = preprocess(window, mu, sd)
                with torch.no_grad():
                    fn_prob = torch.sigmoid(fallnet(tensor)[0, 0]).item()

                res = gate.evaluate(fn_prob, fn_thresh, window)
                fired = res['passed_gate']
                z = res.get('surge_zscore', 0.0)
                bm = res.get('baseline_surge_mean', 0.0)

                tag = "[BASELINE BUILDING]" if not gate._baseline_built else ""
                status = "*** ALARM ***" if fired else "calm"

                print("Frame %4d  FN=%.3f  energy=%.3f  surge=%.3f  z=%.1f  base=%.3f  -> %s %s" % (
                    frame_idx, fn_prob, res['motion_energy'], res['peak_impact'], z, bm, status, tag))

                if fired:
                    false_alarms += 1
                    print("  Suppressed reason: %s" % res.get('suppressed_reason', 'none'))

elapsed = time.time() - start
ser.close()

print()
print("=" * 60)
print("RESULT: %d false alarm(s) in %d frames over %.0f seconds" % (false_alarms, frame_idx, elapsed))
if false_alarms == 0:
    print("PASS: System is stable. Zero false positives.")
else:
    print("FAIL: Still firing incorrectly. Need further investigation.")
