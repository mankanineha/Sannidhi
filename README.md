# Sannidhi

Sannidhi is a contactless Wi-Fi CSI (Channel State Information) fall detection and voice emergency response system. 

It leverages ESP32 microcontrollers to capture Wi-Fi CSI data and processes it using deep learning models (FallNet and SpectraNet) coupled with an Adaptive Z-Score Physics Gate to accurately detect falls while eliminating false positives from environmental noise. 

In the event of a detected fall, an AI voice triage agent interacts with the senior resident, and critical alerts are routed to caregivers via Twilio/WhatsApp integrations.

## Architecture Highlights
- **Sensors:** ESP32 (Arduino / ESP-IDF)
- **Physics Gate:** Adaptive Z-score baseline surge detection
- **Deep Learning:** Dual models (FallNet & SpectraNet)
- **Voice Agent:** Gemini-powered emergency response and triage
- **Alerting:** Caregiver WhatsApp alerts via Twilio

## Setup

1. Create a `.env` file based on `.env.example` and fill in your Twilio/API credentials.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Flash the ESP32 using the Arduino IDE (`arduino_csi_radar.ino`) or the provided `flash_esp32.sh` script.
4. Run the validation and dashboard scripts:
   ```bash
   python validate.py
   python dashboard.py
   ```
