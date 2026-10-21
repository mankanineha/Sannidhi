import socket
import os
import time

# Target ESP32 IP & Broadcast
ESP32_IPS = ["10.94.84.171", "10.94.84.255"]
PORT = int(os.getenv("ESP32_PORT", "5555"))  # Radar port

# Frequency: 100 packets per second (100Hz) — matches CSI-Bench
HZ = int(os.getenv("PING_HZ", "100"))
delay = 1.0 / HZ

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
payload = b"Sannidhi_CSI_Radar_Ping"

print(f"🚀 Starting High-Speed CSI Trigger at {HZ}Hz...")
print(f"📡 Targets: {ESP32_IPS}:{PORT}")
print("🛑 Press Ctrl+C to stop.")

try:
    interval = 1.0 / HZ
    next_time = time.perf_counter() + interval
    packet_count = 0
    start_t = time.perf_counter()
    while True:
        for ip in ESP32_IPS:
            sock.sendto(payload, (ip, PORT))
        packet_count += 1
        
        # High-precision drift-compensated sleep
        sleep_duration = next_time - time.perf_counter()
        if sleep_duration > 0.002:
            time.sleep(sleep_duration - 0.001)
        while time.perf_counter() < next_time:
            pass
        next_time += interval
        
        if packet_count % 1000 == 0:
            elapsed = time.perf_counter() - start_t
            actual_hz = packet_count / elapsed
            print(f"📡 Sent {packet_count} pings | Actual Rate: {actual_hz:.2f} Hz")
except KeyboardInterrupt:
    print("\n🛑 Trigger stopped.")
