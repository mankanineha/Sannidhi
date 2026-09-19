"""
Physics-Based Anti-Hallucination Gate — Adaptive Baseline Edition
==================================================================
PROBLEM FOUND (2026-09-19 diagnostic):
  In this specific ESP32 + room deployment, the ABSOLUTE noise floor is:
    FallNet   quiet room: 0.449 – 0.705  (mean=0.568, std=0.076)
    Surge     quiet room: 0.490 – 0.645  (mean=0.553, std=0.069)

  ALL previous absolute thresholds (0.45, 0.65, etc.) sit INSIDE this noise
  band, causing unavoidable false positives regardless of tuning.

SOLUTION — Adaptive Z-Score Gate:
  1. During the first N frames, build a rolling baseline of the room's
     ambient surge level (mean + std).
  2. Only fire when the surge is BOTH:
       a. Above an absolute floor (ensures some physical event occurred)
       b. Z-score > 3.0 above the room's own baseline (ensures it's anomalous
          relative to THIS specific room's normal activity)
  3. Packet-type-aware diffs: Only compare same-packet-type consecutive frames
     (LLTF vs LLTF, HT-LTF vs HT-LTF) to suppress radio switching artifacts.
"""

import time
import numpy as np
from collections import deque

_ACTIVE_THRESHOLD = 2.0    # subcarrier amplitude below which = zero-padded
_MIN_ACTIVE = 80           # frames with < 80 active subcarriers = LLTF-only


def _count_active(amp_row: np.ndarray) -> int:
    return int((amp_row > _ACTIVE_THRESHOLD).sum())


class PhysicsGate:
    def __init__(self,
                 min_motion_energy: float = 0.05,   # loose absolute floor
                 min_impact_surge: float = 0.40,    # loose absolute floor
                 zscore_threshold: float = 3.5,     # standard deviations above baseline
                 baseline_frames: int = 150,        # detector evaluations, not raw CSI frames
                 debounce_required: int = 2,
                 cooldown_sec: float = 90.0):

        self.min_motion_energy = min_motion_energy
        self.min_impact_surge  = min_impact_surge
        self.zscore_threshold  = zscore_threshold
        self.baseline_frames   = baseline_frames
        self.debounce_required = debounce_required
        self.cooldown_sec      = cooldown_sec

        self.debounce_counter  = 0
        self.last_alert_time   = 0.0

        # Rolling baseline of same-type pair diffs
        self._baseline_surge   = deque(maxlen=baseline_frames)
        self._baseline_energy  = deque(maxlen=baseline_frames)
        self._baseline_built   = False

    def _same_type_diffs(self, window: np.ndarray):
        """Return list of inter-frame diffs for same-packet-type consecutive pairs only."""
        # The CSI contract contains 116 amplitude channels.  Using only 64
        # here made `_MIN_ACTIVE == 80` unreachable, so every frame was
        # incorrectly labelled as the same packet type.
        amp = window[:, :116]
        active_counts = np.array([_count_active(amp[i]) for i in range(len(amp))])
        full_mask = active_counts >= _MIN_ACTIVE
        pkt_mean = np.mean(amp, axis=1, keepdims=True) + 1e-6
        norm_amp = amp / pkt_mean
        diffs = []
        for i in range(1, len(norm_amp)):
            if full_mask[i] == full_mask[i - 1]:
                diffs.append(float(np.mean(np.abs(norm_amp[i] - norm_amp[i - 1]))))
        return diffs

    def compute_kinematics(self, window: np.ndarray) -> tuple:
        diffs = self._same_type_diffs(window)
        if len(diffs) < 5:
            return 0.0, 0.0
        recent = diffs[-100:]
        energy = float(np.mean(recent))
        surge  = float(np.max(recent))
        return energy, surge

    def compute_motion_energy(self, window: np.ndarray) -> float:
        energy, _ = self.compute_kinematics(window)
        return energy

    def _update_baseline(self, energy: float, surge: float):
        """Feed a quiet-room reading into the rolling baseline."""
        self._baseline_surge.append(surge)
        self._baseline_energy.append(energy)
        if len(self._baseline_surge) >= 30:
            self._baseline_built = True

    def _zscore_above_baseline(self, value: float, history: deque) -> float:
        """Return z-score of value relative to history. Returns 0 if not enough data."""
        if len(history) < 10:
            return 0.0
        arr = np.array(list(history))
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-6:
            return 0.0
        return (value - mu) / sigma

    def evaluate(self, raw_prob: float, threshold: float, window: np.ndarray) -> dict:
        now = time.time()
        energy, surge = self.compute_kinematics(window)
        in_cooldown = (now - self.last_alert_time) < self.cooldown_sec
        suppressed_reason = None

        # ----------------------------------------------------------------
        # Absolute minimum floor check (rules out completely dead signals)
        # ----------------------------------------------------------------
        model_passed = raw_prob >= threshold
        if not model_passed:
            # Feed into baseline when model is calm (helps build clean baseline)
            self._update_baseline(energy, surge)
            self.debounce_counter = 0
            return {
                "passed_gate": False,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_counter,
                "in_cooldown": in_cooldown,
                "suppressed_reason": None,
                "baseline_surge_mean": np.mean(self._baseline_surge) if self._baseline_surge else 0.0,
                "surge_zscore": 0.0
            }

        # ----------------------------------------------------------------
        # Adaptive Z-Score Gate: is this surge anomalous vs this room's baseline?
        # ----------------------------------------------------------------
        surge_z = self._zscore_above_baseline(surge, self._baseline_surge)
        energy_z = self._zscore_above_baseline(energy, self._baseline_energy)
        baseline_mean = np.mean(self._baseline_surge) if self._baseline_surge else 0.0

        # If baseline not built yet, be conservative
        if not self._baseline_built:
            self._update_baseline(energy, surge)
            suppressed_reason = f"Building baseline ({len(self._baseline_surge)}/{self.baseline_frames} frames)"
            self.debounce_counter = 0
            return {
                "passed_gate": False,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_counter,
                "in_cooldown": in_cooldown,
                "suppressed_reason": suppressed_reason,
                "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }

        # Gate 1: Absolute floor
        if energy < self.min_motion_energy:
            self._update_baseline(energy, surge)
            self.debounce_counter = 0
            suppressed_reason = f"Low energy ({energy:.3f} < {self.min_motion_energy:.3f})"
            return {
                "passed_gate": False,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_counter,
                "in_cooldown": in_cooldown,
                "suppressed_reason": suppressed_reason,
                "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }

        # Gate 1b: impact floor.  This parameter previously existed but was
        # never applied, leaving the z-score alone to accept tiny variations
        # when the baseline variance was very small.
        if surge < self.min_impact_surge:
            self._update_baseline(energy, surge)
            self.debounce_counter = 0
            suppressed_reason = f"Low impact surge ({surge:.3f} < {self.min_impact_surge:.3f})"
            return {
                "passed_gate": False, "motion_energy": energy, "peak_impact": surge,
                "debounce_count": self.debounce_counter, "in_cooldown": in_cooldown,
                "suppressed_reason": suppressed_reason, "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }

        # Gate 2: Z-score check — must be anomalously high vs baseline
        if surge_z < self.zscore_threshold:
            self._update_baseline(energy, surge)  # normal — feed into baseline
            self.debounce_counter = 0
            suppressed_reason = f"Surge z-score {surge_z:.1f} < {self.zscore_threshold:.1f} (baseline mean={baseline_mean:.3f})"
            return {
                "passed_gate": False,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_counter,
                "in_cooldown": in_cooldown,
                "suppressed_reason": suppressed_reason,
                "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }

        # Gate 3: Cooldown
        if in_cooldown:
            remaining = self.cooldown_sec - (now - self.last_alert_time)
            suppressed_reason = f"Post-alert cooldown ({remaining:.1f}s remaining)"
            return {
                "passed_gate": False,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_counter,
                "in_cooldown": True,
                "suppressed_reason": suppressed_reason,
                "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }

        # All checks passed — debounce
        self.debounce_counter += 1
        if self.debounce_counter >= self.debounce_required:
            self.last_alert_time = now
            self.debounce_counter = 0
            return {
                "passed_gate": True,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_required,
                "in_cooldown": False,
                "suppressed_reason": None,
                "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }
        else:
            suppressed_reason = f"Debouncing ({self.debounce_counter}/{self.debounce_required})"
            return {
                "passed_gate": False,
                "motion_energy": energy,
                "peak_impact": surge,
                "debounce_count": self.debounce_counter,
                "in_cooldown": False,
                "suppressed_reason": suppressed_reason,
                "baseline_surge_mean": baseline_mean,
                "surge_zscore": surge_z
            }
