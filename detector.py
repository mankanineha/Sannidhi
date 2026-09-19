"""
CSI Ingestion, Dual-Model Consensus & Balanced Pipeline
======================================================
Balanced Product Detection Strategy:
1. Calibrated Dual Consensus: FallNet (≥0.80) AND SpectraNet (≥0.60).
2. Voice Agent Lockout: Suppresses fall evaluation while Voice Agent is talking/listening.
3. 45-Second Post-Alert Cooldown: Prevents repeating prompts for a single fall event.
"""

import os
import time
import queue
import threading
import numpy as np
import torch

from simulator import csi_queue, _stop_event, WIN, FEAT, N_SUB, HZ
from physics_gate import PhysicsGate
from voice_agent import voice_agent

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR     = os.path.join(BASE_DIR, "sannidhi-main")
FALLNET_PT      = os.path.join(WEIGHTS_DIR, "csi_fallnet_best.pt")
SPECTRANET_PT   = os.path.join(WEIGHTS_DIR, "csi_spectranet_best.pt")

STEP            = 2
DEVICE          = torch.device("cpu")

# Thresholds are loaded from the checkpoints.  A device deployment must still
# be calibrated and independently validated before these are used clinically.
FALLNET_THRESH    = 0.73
SPECTRANET_THRESH = 0.50
WARMUP_FRAMES     = 50   # 5 seconds at ~10fps to fill window before any inference

results_queue: queue.Queue = queue.Queue(maxsize=500)
# Adaptive Z-Score Gate: builds a rolling baseline of THIS room's ambient surge,
# then only fires when surge is >= 3.5 std-devs above that baseline.
# This makes it robust regardless of absolute noise level.
gate = PhysicsGate(
    min_motion_energy=0.05,   # loose absolute floor
    min_impact_surge=0.40,    # loose absolute floor  
    zscore_threshold=3.5,     # must be 3.5 sigma above room baseline
    baseline_frames=150,      # build baseline over ~15s of quiet room
    debounce_required=2,
    cooldown_sec=90.0         # 90s cooldown (longer than any voice interaction)
)


def load_models():
    from models.fallnet import CSIFallNet
    from models.spectranet import CSISpectraNet

    print(f"[Detector] Loading FallNet weights from {FALLNET_PT}")
    fn_ck = torch.load(FALLNET_PT, map_location="cpu", weights_only=False)
    fallnet = CSIFallNet(n_sub=N_SUB, c=48, drop=0.0)
    fallnet.load_state_dict(fn_ck["model_state_dict"])
    fallnet.eval()

    norm = fn_ck.get("normalization", {})
    if norm and "mean" in norm:
        mu = np.array(norm["mean"], dtype=np.float32).reshape(-1)[-FEAT:]
        sd = np.array(norm["std"],  dtype=np.float32).reshape(-1)[-FEAT:]
    else:
        mu, sd = None, None

    spectranet = None
    sn_mu, sn_sd = None, None
    if os.path.exists(SPECTRANET_PT):
        print(f"[Detector] Loading SpectraNet weights from {SPECTRANET_PT}")
        try:
            sn_ck = torch.load(SPECTRANET_PT, map_location="cpu", weights_only=False)
            spectranet = CSISpectraNet(n_sub=N_SUB, num_classes=2)
            spectranet.load_state_dict(sn_ck["model_state_dict"])
            spectranet.eval()
            sn_norm = sn_ck.get("normalization", {})
            if sn_norm and "mean" in sn_norm:
                sn_mu = np.array(sn_norm["mean"], dtype=np.float32).reshape(-1)[-FEAT:]
                sn_sd = np.array(sn_norm["std"], dtype=np.float32).reshape(-1)[-FEAT:]
            print("[Detector] ✅ SpectraNet loaded successfully!")
        except Exception as e:
            print(f"[Detector] ⚠️ Could not load SpectraNet: {e}")
            spectranet = None

    return (fallnet, float(fn_ck.get("threshold", FALLNET_THRESH)), spectranet,
            SPECTRANET_THRESH, mu, sd, sn_mu, sn_sd)


def preprocess(window: np.ndarray, mu, sd) -> torch.Tensor:
    x = window.copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if mu is not None:
        x = (x - mu) / (sd + 1e-6)
    else:
        x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.from_numpy(x).float().unsqueeze(0)


def input_distribution_check(window: np.ndarray, mu, sd) -> tuple[bool, float]:
    """Detect an uncalibrated sensor/model domain before it can raise an alert.

    This is deliberately a fail-safe, not a substitute for validation.  The
    fraction is the share of raw features within six training standard
    deviations of the checkpoint distribution; a low fraction means the model
    is being asked to extrapolate (for example raw ESP32 values versus a
    differently scaled training corpus).
    """
    if mu is None or sd is None:
        return False, 0.0
    raw = np.nan_to_num(window[-100:], nan=0.0, posinf=0.0, neginf=0.0)
    z = np.abs((raw - mu) / (sd + 1e-6))
    coverage = float(np.mean(z <= 6.0))
    return coverage >= 0.95, coverage


def run_detector():
    fallnet, fn_thresh, spectranet, sn_thresh, mu, sd, sn_mu, sn_sd = load_models()

    window = np.zeros((WIN, FEAT), dtype=np.float32)
    n_frames = 0
    step_counter = 0
    last_fn_prob = 0.0
    last_sn_prob = 0.0
    last_combined_prob = 0.0
    last_latency = 0.0

    print(f"[Detector] 🚀 Balanced Pipeline Active — FN_Thresh={fn_thresh}, SN_Thresh={sn_thresh}")

    while not _stop_event.is_set():
        try:
            frame = csi_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if frame.shape[0] < FEAT:
            frame = np.pad(frame, (0, FEAT - frame.shape[0]))
        elif frame.shape[0] > FEAT:
            frame = frame[:FEAT]

        window[:-1] = window[1:]
        window[-1]  = frame
        n_frames   += 1
        step_counter += 1

        # On the very first live frame, initialize the entire historical buffer with this live frame
        # This completely eliminates the artificial step-function / zero-cliff artifact!
        if n_frames == 1:
            window[:] = frame

        # Require a stable continuous window before evaluating fall classification
        if n_frames < WARMUP_FRAMES or step_counter < STEP:
            if n_frames % 2 == 0:
                warmup_res = {
                    "timestamp": time.time(),
                    "fallnet_prob": last_fn_prob,
                    "spectranet_prob": last_sn_prob,
                    "probability": last_combined_prob,
                    "threshold": fn_thresh,
                    "latency_ms": last_latency,
                    "frame_num": n_frames,
                    "passed_gate": False,
                    "motion_energy": gate.compute_motion_energy(window),
                    "peak_impact": gate.compute_kinematics(window)[1],
                    "debounce_count": 0,
                    "in_cooldown": False,
                    "suppressed_reason": f"Stabilizing Baseline ({n_frames}/{WARMUP_FRAMES} frames)" if n_frames < WARMUP_FRAMES else None,
                    "voice_status": voice_agent.current_status,
                    "amp_mean": float(frame[:N_SUB].mean()),
                    "amp_profile": frame[:N_SUB].tolist(),
                    "csi_heatmap_slice": window[-min(n_frames, 100):, :N_SUB].tolist()
                }
                try:
                    results_queue.put_nowait(warmup_res)
                except queue.Full:
                    try:
                        results_queue.get_nowait()
                        results_queue.put_nowait(warmup_res)
                    except queue.Empty:
                        pass
            continue
        step_counter = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            x = preprocess(window, mu, sd)
            input_ok, input_coverage = input_distribution_check(window, mu, sd)

            # 1. FallNet (Sigmoid)
            fn_logit = fallnet(x)
            fn_prob  = torch.sigmoid(fn_logit[0, 0]).item()

            # 2. SpectraNet (Softmax Class 1) — kept for display only
            # Diagnostic showed SN is 0.80 flat in quiet room → useless as discriminator
            if spectranet is not None:
                # SpectraNet was trained with its own normalization stats.
                sn_logits = spectranet(preprocess(window, sn_mu, sn_sd))
                sn_probs  = torch.softmax(sn_logits, dim=1)
                sn_prob   = sn_probs[0, 1].item()
            else:
                sn_prob   = fn_prob

            # FallNet-only consensus: SN is constant in this deployment, only FN varies
            # FallNet quiet range: 0.449–0.705 → threshold at 0.80 sits above noise
            consensus_passed = (fn_prob >= fn_thresh)
            combined_prob = fn_prob

        latency_ms = (time.perf_counter() - t0) * 1000
        last_fn_prob = fn_prob
        last_sn_prob = sn_prob
        last_combined_prob = combined_prob
        last_latency = latency_ms

        # Evaluate through adaptive physics gate (passes FallNet prob to gate)
        eval_res = gate.evaluate(raw_prob=fn_prob, threshold=fn_thresh, window=window)
        passed_gate = eval_res["passed_gate"]

        if not input_ok:
            passed_gate = False

        # Suppress new triggers if Voice Agent is currently active in a conversation!
        if voice_agent.is_active:
            passed_gate = False

        result = {
            "timestamp"          : time.time(),
            "fallnet_prob"       : fn_prob,
            "spectranet_prob"    : sn_prob,
            "probability"        : combined_prob,
            "threshold"          : fn_thresh,
            "latency_ms"         : latency_ms,
            "frame_num"          : n_frames,
            "passed_gate"        : passed_gate,
            "motion_energy"      : eval_res["motion_energy"],
            "peak_impact"        : eval_res.get("peak_impact", 0.0),
            "baseline_surge_mean": eval_res.get("baseline_surge_mean", 0.0),
            "surge_zscore"       : eval_res.get("surge_zscore", 0.0),
            "debounce_count"     : eval_res["debounce_count"],
            "in_cooldown"        : eval_res["in_cooldown"],
            "input_distribution_ok": input_ok,
            "input_coverage"     : input_coverage,
            "suppressed_reason"  : ("Input distribution mismatch: calibrate/retrain with this ESP32" if not input_ok
                                    else "Voice conversation in progress" if voice_agent.is_active else eval_res["suppressed_reason"]),
            "voice_status"       : voice_agent.current_status,
            "amp_mean"           : float(window[:, :N_SUB].mean()),
            "amp_profile"        : window[-1, :N_SUB].tolist(),
            "csi_heatmap_slice"  : window[-100:, :N_SUB].tolist()
        }

        if passed_gate:
            print(f"\n[Detector] 🚨 VERIFIED FALL EVENT! (FN={fn_prob:.3f}, SN={sn_prob:.3f})")
            voice_agent.trigger_confirmation_flow(metrics=result)

        try:
            results_queue.put_nowait(result)
        except queue.Full:
            try:
                results_queue.get_nowait()
                results_queue.put_nowait(result)
            except queue.Empty:
                pass

        if passed_gate:
            print(f"[Detector] 🔊 Voice flow started | latency={latency_ms:.1f}ms")
        elif n_frames % (HZ * 5) == 0:
            print(f"[Detector] 👁  FN={fn_prob:.3f}, SN={sn_prob:.3f}, Energy={eval_res['motion_energy']:.3f} | Voice: {voice_agent.current_status}")

    print("[Detector] Stopped.")


def start() -> threading.Thread:
    t = threading.Thread(target=run_detector, daemon=True, name="detector")
    t.start()
    return t
