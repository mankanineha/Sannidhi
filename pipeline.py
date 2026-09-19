"""
Sannidhi Fall Detection Pipeline — Main Entry Point
=====================================================
Wired components: simulator → detector → dashboard → voice_agent → caregiver_alert

Usage:
    python pipeline.py                   # replay real CSI + live dashboard
    python pipeline.py --mode synthetic  # synthetic CSI + live dashboard
    python pipeline.py --mode hardware   # live ESP32 serial input + live dashboard
    python pipeline.py --headless        # no GUI, print to terminal
    python pipeline.py --test            # 10-second self-test
"""

import argparse
import sys
import os
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import simulator
import detector


def run_headless(mode: str, duration: float = None):
    print("=" * 60)
    print("  Sannidhi Fall Detection Pipeline (Headless)")
    print("=" * 60)

    from detector import results_queue
    import queue

    print(f"[Main] Starting simulator in '{mode}' mode...")
    sim_thread = simulator.start(mode, verbose=True)

    print("[Main] Starting detector...")
    det_thread = detector.start()

    start_time = time.time()
    fall_count = 0

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                print(f"\n[Main] Duration {duration}s reached. Stopping.")
                break

            try:
                r = results_queue.get(timeout=1.0)
                if r.get("passed_gate", False):
                    fall_count += 1
                    print(f"\n{'='*50}")
                    print(f"🚨 FALL #{fall_count} DETECTED at frame {r['frame_num']}")
                    print(f"   Probability  : {r['probability']:.4f}")
                    print(f"   Threshold    : {r['threshold']:.2f}")
                    print(f"   Latency      : {r['latency_ms']:.1f} ms")
                    print(f"{'='*50}\n")
                elif r["frame_num"] % 400 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:6.1f}s] Frame {r['frame_num']:>6} | "
                          f"P(fall)={r['probability']:.3f} | "
                          f"latency={r['latency_ms']:.1f}ms")
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    simulator.stop()
    print(f"\n[Main] Done. Total falls detected: {fall_count}")


def run_with_dashboard(mode: str):
    print("=" * 60)
    print("  Sannidhi Fall Detection Pipeline")
    print(f"  Mode: {mode}")
    print("=" * 60)

    print(f"[Main] Starting ESP32 simulator in '{mode}' mode...")
    sim_thread = simulator.start(mode, verbose=False)
    time.sleep(0.5)

    print("[Main] Starting fall detector...")
    det_thread = detector.start()
    time.sleep(0.5)

    print("[Main] Launching dashboard...")
    print("[Main] 💡 TIP: Click 'Simulate Fall' button to simulate a fall event")
    print("[Main] 💡 TIP: Press Ctrl+C in terminal to quit\n")

    import dashboard
    ani = dashboard.start(sim_mode=mode)


def run_selftest():
    print("=" * 60)
    print("  Sannidhi Self-Test (10 seconds)")
    print("=" * 60)

    sim_thread = simulator.start("replay", verbose=False)
    det_thread = detector.start()

    from detector import results_queue
    import queue

    print("[Test] Waiting for first inference result...")
    found = False
    deadline = time.time() + 15

    while time.time() < deadline:
        try:
            r = results_queue.get(timeout=1.0)
            print(f"[Test] ✅ First result received!")
            print(f"       Frame     : {r['frame_num']}")
            print(f"       P(fall)   : {r['probability']:.4f}")
            print(f"       Threshold : {r['threshold']:.2f}")
            print(f"       Latency   : {r['latency_ms']:.1f} ms")
            found = True
            break
        except queue.Empty:
            print("[Test] Waiting...", flush=True)

    if not found:
        print("[Test] ❌ No result received in 15 seconds.")
        simulator.stop()
        sys.exit(1)

    simulator.stop()
    print("\n" + "=" * 60)
    print("✅ SELF-TEST PASSED — Pipeline verified!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Sannidhi Real-Time Fall Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=["replay", "synthetic", "hardware"], default="replay",
        help="Data source: 'replay' streams raw_capture.csv, 'synthetic' generates fake CSI, 'hardware' connects live to ESP32 USB"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI dashboard"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run 10-second self-test and exit"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="(headless only) Stop after N seconds"
    )

    args = parser.parse_args()

    if args.test:
        run_selftest()
    elif args.headless:
        run_headless(args.mode, args.duration)
    else:
        run_with_dashboard(args.mode)


if __name__ == "__main__":
    main()
