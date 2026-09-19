import socket
import time

# --- UPDATE THIS TO YOUR ESP32's CURRENT IP ON YOUR PHONE ---
ESP32_IP = "192.168.235.188" 
PORT = 5555 # Dummy port, the ESP32 just needs to "hear" the wave

# Frequency: 100 packets per second (100Hz) — matches CSI-Bench
HZ = 100
delay = 1.0 / HZ

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
payload = b"Sannidhi_CSI_Radar_Ping"

print(f"🚀 Starting High-Speed CSI Trigger at {HZ}Hz...")
print(f"📡 Target: {ESP32_IP}")
print("🛑 Press Ctrl+C to stop.")

try:
    while True:
        # Blasting the invisible waves
        sock.sendto(payload, (ESP32_IP, PORT))
        time.sleep(delay)
except KeyboardInterrupt:
    print("\n🛑 Trigger stopped.")