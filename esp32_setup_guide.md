# ESP32 Setup & Live Integration Guide for Sannidhi

This guide details all the exact steps to connect your ESP32 hardware to your laptop (or Raspberry Pi 5) and run live Wi-Fi CSI fall detection.

---

## Step 1: Physical Connection 🔌
1. Take your **ESP32 DevKit** board.
2. Plug the **Micro-USB** end into the ESP32 and the **USB-C** end into your Mac / Raspberry Pi 5.
3. Observe the red/blue power LED turn ON on the ESP32.

---

## Step 2: Find the USB Serial Port on Mac 🔍
Open Terminal and run:
```bash
ls /dev/cu.*
```
Look for a device named like one of these:
- `/dev/cu.usbserial-1410`
- `/dev/cu.wchusbserial1420`
- `/dev/cu.SLAB_USBtoUART`

*(If no USB device appears, install the CH340 or CP210x Mac USB driver for your ESP32 board).*

---

## Step 3: Flash the Firmware to ESP32 (One-Time Setup) ⚡
Your project already contains the custom C code for CSI tensor extraction in `sannidhi-main/main/scan.c`.

1. Open `sannidhi-main/main/scan.c` in an editor.
2. Lines 88–89 set the home Wi-Fi credentials:
   ```c
   .ssid = "Rushu-TP",
   .password = "56781234"
   ```
3. Flash the code to the ESP32 using ESP-IDF or esptool:
   ```bash
   cd sannidhi-main
   idf.py -p /dev/cu.usbserial-1410 flash
   ```
4. Once flashed, the ESP32 connects to your Wi-Fi router and starts transmitting CSI amplitude + phase tensors over USB at 100 Hz (`Baud: 921600`).

---

## Step 4: Run the Live Pipeline with ESP32 🚀
Simply run:

```bash
python3 pipeline.py --mode hardware
```

### What happens automatically:
1. Sannidhi auto-detects your connected ESP32 on `/dev/cu.usbserial*`.
2. Live Wi-Fi CSI tensors flow from the room into `simulator.csi_queue`.
3. Dual AI Models (**CSI-FallNet** & **CSISpectraNet**) process the live Wi-Fi signal in real time.
4. The **Physics Gate** verifies genuine physical movement.
5. If a fall occurs, the **Voice Agent** speaks through the laptop speakers/mic and confirms if the senior citizen is okay.
6. If the person is unresponsive or says "No", a **WhatsApp Alert** is instantly dispatched to the caregiver's phone!
