import serial
import time

# --- CONFIGURATION ---
PORT = "COM5"
BAUD = 921600
FILENAME = "raw_capturee.csv"
WINDOW_SIZE = 500

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    # Give the serial port a second to wake up
    time.sleep(2)
    ser.flushInput() 

    with open(FILENAME, "w", encoding="utf-8") as f:
        print(f"📡 Sannidhi Radar Recording... (Target: {PORT})")
        print("🛑 Press Ctrl+C to save and stop.\n")
        
        count = 0
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Clean and save only the TENSOR rows
                if "CSI_TENSOR" in line:
                    # Clean off leading dots or garbage
                    start_idx = line.find("CSI_TENSOR")
                    clean_line = line[start_idx:]
                    
                    f.write(clean_line + "\n")
                    count += 1
                    
                    # Print every 10th row so the screen isn't too fast
                    if count % 10 == 0:
                        print(f"✅ Captured {count} rows...")

                    # Print window progress every time a full window is collected
                    if count % WINDOW_SIZE == 0:
                        window_count = count // WINDOW_SIZE
                        print(f"🔲 Window {window_count} complete ({count} frames)")

except KeyboardInterrupt:
    completed_windows = count // WINDOW_SIZE
    print(f"\n\n💾 DONE! {count} clean rows saved to {FILENAME} ({completed_windows} complete windows)")
except Exception as e:
    print(f"❌ Error: {e}. (Make sure NO other terminal is using {PORT})")
finally:
    if 'ser' in locals():
        ser.close()