import serial

port = "/dev/cu.usbserial-0001"
baud = 921600

print(f"Testing raw serial connection to {port} at {baud} baud...")
try:
    ser = serial.Serial(port, baud, timeout=2)
    print("Port opened! Waiting for data...")
    valid = 0
    for i in range(20):
        text = ser.readline().decode("utf-8", errors="replace").strip()
        if not text:
            continue
        fields = text.split(",")
        if text.startswith("CSI_TENSOR,") and len(fields) == 235:
            valid += 1
            print(f"VALID CSI ROW {valid}: 235 fields; starts {text[:55]}...")
            if valid == 3:
                break
        else:
            print(f"Non-CSI/incomplete row: {text[:100]}")
    if valid == 0:
        print("FAIL: no complete CSI rows received.")
    else:
        print(f"PASS: {valid} complete CSI row(s) received.")
except Exception as e:
    print(f"Error: {e}")
