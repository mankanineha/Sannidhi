import serial
import json
import re
from csi_processor_paper import CSIProcessorCSIBench

# From the image provided: The ESP32 is on COM5
SERIAL_PORT = "COM5" 
BAUD_RATE = 115200 # Standard for ESP-CSI
# The paper segments fall detection into 5-second intervals. 
# Typical sampling rate is 100Hz for general sensing tasks (Section 3.2).
processor = CSIProcessorCSIBench(sampling_rate=100, window_seconds=5, subcarriers=64)

def read_esp32_serial():
    print(f"Connecting to ESP32 on {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Successfully connected! Listening for CSI data...")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # ESP-CSI typically outputs lines starting with "CSI_DATA" followed by comma-separated values
            if "CSI_DATA" in line:
                # Extract the array of integers
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    try:
                        raw_iq_str = match.group(1)
                        raw_iq_data = [int(x) for x in raw_iq_str.split(',')]
                        
                        # Process using the exact CSI-Bench methodology
                        normalized_csi = processor.process_raw_esp32_csi(raw_iq_data)
                        
                        if normalized_csi:
                            # This is a fully preprocessed 64-subcarrier packet ready for the model
                            # The model expects a (64, T) or similar matrix.
                            print(f"Received & Normalized Packet: {normalized_csi[:3]}...") # Print first 3 SCs
                        else:
                            print("Buffering data... waiting to fill the 5-second (500 packet) window.")
                    except ValueError:
                        pass # Ignore malformed lines
                        
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        print("Please ensure no other programs (like Arduino IDE Serial Monitor) are using the port.")

if __name__ == "__main__":
    read_esp32_serial()
