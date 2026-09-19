"""
Sannidhi V2 — Real-Time Fall Detection Dashboard
=================================================
Live GUI displaying:
  - Dual model fall probabilities (FallNet vs SpectraNet)
  - Anti-hallucination physics gate state (Debounce N/3, Energy, Cooldown)
  - Voice confirmation agent status
  - Real-time CSI amplitude waveform + per-subcarrier profile
  - Interactive triggers: [Simulate Fall] & [Quit]
"""

import sys
import os
import time
import queue
import threading
import numpy as np
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from collections import deque

from detector import results_queue
import simulator

HISTORY = 200

fn_prob_history = deque([0.0] * HISTORY, maxlen=HISTORY)
sn_prob_history = deque([0.0] * HISTORY, maxlen=HISTORY)
amp_history     = deque([0.0] * HISTORY, maxlen=HISTORY)
gate_history    = deque([False] * HISTORY, maxlen=HISTORY)

last_result = {
    "probability": 0.0, "fallnet_prob": 0.0, "spectranet_prob": 0.0,
    "threshold": 0.73, "latency_ms": 0.0, "frame_num": 0,
    "passed_gate": False, "motion_energy": 0.0, "debounce_count": 0,
    "in_cooldown": False, "suppressed_reason": None,
    "voice_status": "Idle", "amp_profile": [0.0] * 116, "amp_mean": 0.0
}
fall_count  = 0
alert_until = 0.0


def update_state():
    global fall_count, alert_until, last_result
    changed = False
    while True:
        try:
            r = results_queue.get_nowait()
            fn_prob_history.append(r.get("fallnet_prob", r["probability"]))
            sn_prob_history.append(r.get("spectranet_prob", r["probability"]))
            amp_history.append(r["amp_mean"])
            gate_history.append(r.get("passed_gate", False))

            last_result = r
            if r.get("passed_gate", False):
                fall_count += 1
                alert_until = time.time() + 4.0
            changed = True
        except queue.Empty:
            break
    return changed


fig = plt.figure(figsize=(14, 9.5), facecolor="#0d0d1a")
fig.canvas.manager.set_window_title("Sannidhi V2 — Real-Time WiFi Fall Detection")

gs = fig.add_gridspec(
    4, 2,
    height_ratios=[0.7, 2.4, 1.6, 0.5],
    width_ratios=[3, 1.2],
    hspace=0.45, wspace=0.3,
    left=0.08, right=0.95, top=0.92, bottom=0.08
)

ax_status   = fig.add_subplot(gs[0, :])
ax_waveform = fig.add_subplot(gs[1, 0])
ax_prob_bar = fig.add_subplot(gs[1, 1])
ax_profile  = fig.add_subplot(gs[2, 0])
ax_info     = fig.add_subplot(gs[2, 1])
ax_btn      = fig.add_subplot(gs[3, 0])

DARK_BG  = "#0d0d1a"
CARD_BG  = "#12122a"
ACCENT   = "#7c3aed"
CYAN     = "#06b6d4"
PURPLE   = "#a855f7"
GREEN    = "#10b981"
RED      = "#ef4444"
ORANGE   = "#f59e0b"
TEXT_CLR = "#e2e8f0"
GRID_CLR = "#1e1e3a"

for ax in [ax_waveform, ax_prob_bar, ax_profile, ax_info, ax_status, ax_btn]:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)

ax_btn.set_facecolor(DARK_BG)
ax_btn.axis("off")

ax_status.axis("off")
status_patch = mpatches.FancyBboxPatch(
    (0.01, 0.05), 0.98, 0.90,
    boxstyle="round,pad=0.02",
    facecolor=CARD_BG, edgecolor=ACCENT, linewidth=2,
    transform=ax_status.transAxes, clip_on=False
)
ax_status.add_patch(status_patch)
status_txt = ax_status.text(
    0.5, 0.55, "● MONITORING  —  Normal Activity",
    transform=ax_status.transAxes,
    fontsize=17, fontweight="bold",
    color=GREEN, ha="center", va="center",
    fontfamily="monospace"
)
title_txt = ax_status.text(
    0.5, 0.92,
    "Sannidhi V2  |  FallNet + Independent SpectraNet Metrics  |  Safety Gate Active",
    transform=ax_status.transAxes,
    fontsize=9, color="#64748b", ha="center", va="top"
)

ax_waveform.set_title("CSI Signal & Dual Model Probabilities", color=TEXT_CLR, fontsize=10, pad=4)
ax_waveform.set_xlabel("Time (frames)", color=TEXT_CLR, fontsize=8)
ax_waveform.set_ylabel("Amplitude / Prob", color=TEXT_CLR, fontsize=8)
ax_waveform.set_xlim(0, HISTORY)
ax_waveform.set_ylim(-5, 65)
ax_waveform.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.7)

xs = list(range(HISTORY))
wf_line,     = ax_waveform.plot(xs, list(amp_history), color="#38bdf8", linewidth=1.2, label="CSI Amp")
fn_line,     = ax_waveform.plot(xs, list(fn_prob_history), color=CYAN, linewidth=1.2, linestyle="--", label="FallNet P(fall)×50")
sn_line,     = ax_waveform.plot(xs, list(sn_prob_history), color=PURPLE, linewidth=1.2, linestyle=":", label="SpectraNet P(fall)×50")
ax_waveform.legend(loc="upper right", fontsize=7, facecolor=CARD_BG, labelcolor=TEXT_CLR, framealpha=0.8, edgecolor=GRID_CLR)

ax_prob_bar.set_title("Dual Model Consensus", color=TEXT_CLR, fontsize=10, pad=4)
ax_prob_bar.set_xlim(0, 1)
ax_prob_bar.set_ylim(0, 1)
ax_prob_bar.set_xticks([])
ax_prob_bar.set_yticks(np.arange(0, 1.1, 0.1))
ax_prob_bar.yaxis.set_tick_params(labelsize=7)
ax_prob_bar.grid(True, axis="y", color=GRID_CLR, linewidth=0.5, alpha=0.7)

# Background track containers
ax_prob_bar.bar([0.3, 0.7], [1.0, 1.0], width=0.3, color="#1e1e38", edgecolor="#2d2d55", linewidth=1.0, align="center")

bar_fn = ax_prob_bar.bar(0.3, 0.02, width=0.3, color=CYAN, edgecolor="#67e8f9", linewidth=1.2, align="center")[0]
bar_sn = ax_prob_bar.bar(0.7, 0.02, width=0.3, color=PURPLE, edgecolor="#c084fc", linewidth=1.2, align="center")[0]
thr_line = ax_prob_bar.axhline(0.73, color=RED, linewidth=1.5, linestyle="--")

lbl_fn = ax_prob_bar.text(0.3, 0.06, "FN\n0.00", ha="center", fontsize=7.5, color=CYAN, transform=ax_prob_bar.transAxes, fontweight="bold")
lbl_sn = ax_prob_bar.text(0.7, 0.06, "SN\n0.00", ha="center", fontsize=7.5, color=PURPLE, transform=ax_prob_bar.transAxes, fontweight="bold")

ax_profile.set_title("Subcarrier Spectrum (116 Subcarriers)", color=TEXT_CLR, fontsize=9, pad=4)
ax_profile.set_xlabel("Subcarrier index", color=TEXT_CLR, fontsize=8)
ax_profile.set_ylabel("Amplitude", color=TEXT_CLR, fontsize=8)
ax_profile.set_xlim(0, 116)
ax_profile.set_ylim(0, 75)
ax_profile.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.5)
profile_line, = ax_profile.plot(range(116), [0] * 116, color="#a78bfa", linewidth=1.0)
profile_fill  = ax_profile.fill_between(range(116), [0] * 116, color="#a78bfa", alpha=0.15)

ax_info.axis("off")
info_txt = ax_info.text(
    0.05, 0.95, "Initialising Sannidhi V2...",
    transform=ax_info.transAxes,
    fontsize=8.0, color=TEXT_CLR, va="top", fontfamily="monospace",
    linespacing=1.6
)

btn_ax_fall = fig.add_axes([0.10, 0.01, 0.18, 0.045])
btn_fall = Button(btn_ax_fall, "Simulate Fall", color=CARD_BG, hovercolor="#7c1d1d")
btn_fall.label.set_color(TEXT_CLR)
btn_fall.label.set_fontsize(9)

btn_ax_quit = fig.add_axes([0.32, 0.01, 0.14, 0.045])
btn_quit = Button(btn_ax_quit, "Quit Pipeline", color=CARD_BG, hovercolor="#1e1e3a")
btn_quit.label.set_color(TEXT_CLR)
btn_quit.label.set_fontsize(9)

mode_txt = fig.text(0.55, 0.022, "Mode: replay", fontsize=8, color="#64748b", fontfamily="monospace")
falls_txt = fig.text(0.78, 0.022, "Falls: 0", fontsize=8, color=RED, fontfamily="monospace", fontweight="bold")


def animate(frame_idx):
    global fall_count, profile_fill

    update_state()
    r = last_result
    now = time.time()

    fn_p = r.get("fallnet_prob", r["probability"])
    sn_p = r.get("spectranet_prob", r["probability"])
    passed = r.get("passed_gate", False)
    coverage = r.get("input_coverage", 0.0)
    input_ok = r.get("input_distribution_ok", False)

    wf_line.set_ydata(list(amp_history))
    fn_line.set_ydata([p * 50 for p in fn_prob_history])
    sn_line.set_ydata([p * 50 for p in sn_prob_history])

    bar_fn.set_height(max(fn_p, 0.02))
    bar_sn.set_height(max(sn_p, 0.02))
    lbl_fn.set_text(f"FN\n{fn_p:.2f}")
    lbl_sn.set_text(f"SN\n{sn_p:.2f}")

    p_data = r.get("amp_profile", [0.0] * 116)
    profile_line.set_ydata(p_data)
    profile_fill.remove()
    profile_fill = ax_profile.fill_between(range(116), p_data, color="#a78bfa", alpha=0.15)

    alerting = now < alert_until
    if not input_ok:
        status_txt.set_text("⚠ CALIBRATION REQUIRED — ALERTS SAFELY BLOCKED")
        status_txt.set_color(ORANGE)
        status_patch.set_facecolor(ORANGE)
        status_patch.set_alpha(0.25)
    elif alerting or passed:
        status_txt.set_text("🚨 GENUINE FALL DETECTED — VOICE PIPELINE ACTIVE!")
        status_txt.set_color(RED)
        status_patch.set_facecolor(RED)
        status_patch.set_alpha(0.3)
    else:
        status_txt.set_text("● MONITORING — Environment Normal")
        status_txt.set_color(GREEN)
        status_patch.set_facecolor(CARD_BG)
        status_patch.set_alpha(1.0)

    thr = r.get("threshold", 0.70)
    thr_line.set_ydata([thr, thr])

    energy = r.get("motion_energy", 0.0)
    peak_impact = r.get("peak_impact", 0.0)
    deb    = r.get("debounce_count", 0)
    cooldown = r.get("in_cooldown", False)
    voice  = r.get("voice_status", "Idle")
    suppressed = r.get("suppressed_reason", None)

    info_str = (
        f"━━ MODEL METRICS ━━━━━━━\n"
        f"FallNet P(fall)  : {fn_p:.3f}\n"
        f"SpectraNet P     : {sn_p:.3f}\n"
        f"Threshold        : {thr:.2f}\n"
        f"Inference Latency: {r['latency_ms']:.1f} ms\n\n"
        f"Input Match      : {coverage*100:.1f}% ({'OK' if input_ok else 'BLOCKED'})\n\n"
        f"━━ PHYSICS GATE (Adaptive) ━━\n"
        f"Motion Energy    : {energy:.3f}\n"
        f"Impact Surge     : {peak_impact:.3f}\n"
        f"Baseline Surge   : {r.get('baseline_surge_mean', 0.0):.3f}\n"
        f"Surge Z-Score    : {r.get('surge_zscore', 0.0):.1f}σ  (need >3.5)\n"
        f"Cooldown Active  : {'YES' if cooldown else 'No'}\n"
        f"State            : {suppressed if suppressed else ('Active' if not cooldown else 'Cooldown')}\n\n"
        f"━━ VOICE AGENT ━━━━━━━━━\n"
        f"Status: {voice}"
    )
    info_txt.set_text(info_str)
    falls_txt.set_text(f"Verified Falls: {fall_count}")

    return (wf_line, fn_line, sn_line, bar_fn, bar_sn, thr_line, profile_line, status_txt, info_txt)


def on_inject_fall(event):
    simulator.trigger_fall()
    print("[Dashboard] 💥 Simulated Fall Triggered!")

def on_quit(event):
    simulator.stop()
    plt.close("all")
    sys.exit(0)

btn_fall.on_clicked(on_inject_fall)
btn_quit.on_clicked(on_quit)


def start(sim_mode="replay"):
    mode_txt.set_text(f"Mode: {sim_mode}")
    ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return ani
