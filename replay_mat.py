"""
Replay CSI .mat file to Sannidhi Dashboard over UDP
===================================================
Simulates the ESP32 radar streaming a ground-truth CSI sample
(e.g., 3-1-7-10.mat) directly to localhost:5555 at 100Hz.
"""

import sys
import time
import socket
from pathlib import Path
import scipy.io as sio

MAT_FILE = Path(__file__).parent / "3-1-7-10.mat"
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5555
RATE_HZ = 100

def replay():
    if not MAT_FILE.exists():
        print(f"❌ Error: {MAT_FILE} not found!")
        return

    data = sio.loadmat(str(MAT_FILE))['CSIamp']
    if data.shape == (52, 950):
        data = data.T

    total_frames = data.shape[0]
    print("=" * 60)
    print("📡 Sannidhi CSI .mat Playback Simulator")
    print("=" * 60)
    print(f"📁 File: {MAT_FILE.name}")
    print(f"📊 Dimensions: {data.shape} ({total_frames} frames = {total_frames/RATE_HZ:.1f}s)")
    print(f"🚀 Streaming to: {TARGET_HOST}:{TARGET_PORT} at {RATE_HZ}Hz...")
    print("🛑 Press Ctrl+C to stop.\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    delay = 1.0 / RATE_HZ

    loop_count = 0
    try:
        while True:
            loop_count += 1
            print(f"▶️ [Loop {loop_count}] Streaming {total_frames} frames of fall event...")
            for idx in range(total_frames):
                feats = data[idx]
                feat_str = ",".join(f"{v:.2f}" for v in feats)
                packet = f"CSI_TENSOR,{int(time.time()*1000)},{idx+1},{feat_str}\n"
                sock.sendto(packet.encode("utf-8"), (TARGET_HOST, TARGET_PORT))
                time.sleep(delay)
            print(f"✅ Loop {loop_count} complete. Pausing 2s before repeating...\n")
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n🛑 Playback stopped.")

if __name__ == "__main__":
    replay()
