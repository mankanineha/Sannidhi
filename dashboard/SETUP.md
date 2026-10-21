# CSI-Bench Compatible Fall Detection Dashboard Setup

This guide will walk you through setting up a safe, isolated Python environment, installing the necessary libraries, and running both the simulated dashboard and the live ESP32 hardware bridge.

---

## Step 1: Create a Python Virtual Environment
To avoid installing libraries directly onto your system (and keep your workspace clean), we will use a **Virtual Environment**.

1. Open **Command Prompt** or **PowerShell**.
2. Navigate to your dashboard folder:
   ```cmd
   cd E:\sannidhi-main\sannidhi-main\dashboard
   ```
3. Create a virtual environment named `venv`:
   ```cmd
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```cmd
   .\venv\Scripts\activate
   ```
   *(You should now see `(venv)` at the start of your command line prompt. This means you are safely inside the isolated environment).*

---

## Step 2: Install Required Libraries
With the virtual environment activated, install the required packages:

```cmd
pip install fastapi uvicorn numpy websockets pyserial onnxruntime
```

**Library breakdown:**
- `onnxruntime`: Loads and runs your `csi_spectranet_best.onnx` model weights at high speed.
- `fastapi` & `uvicorn`: Runs the high-speed backend web server.
- `numpy`: Handles the complex CSI-Bench math (Matrix normalization, Amplitude extraction).
- `websockets`: Streams the real-time data to your HTML dashboard.
- `pyserial`: Allows Python to read the live CSI data from your ESP32 on `COM5`.

---

## Step 3: Run the Live Dashboard
The backend automatically detects a single USB serial ESP32 port. To force a specific port, set `SANNIDHI_SERIAL_PORT`, for example in PowerShell:

```powershell
$env:SANNIDHI_SERIAL_PORT = "COM7"
```

It ignores Bluetooth virtual ports. If more than one USB serial port is connected, set the variable explicitly.

1. Ensure your ESP32 is plugged in and no other programs (like the Arduino IDE) are using the serial port.
2. Ensure your virtual environment is still activated `(venv)`.
3. Navigate into the backend folder:
   ```cmd
   cd backend
   ```
4. Start the FastAPI server:
   ```cmd
   python app.py
   ```
5. You will see terminal output confirming:
   - `Successfully loaded best model: ../../csi_spectranet_best.onnx`
   - `SUCCESS: Connected to live ESP32 hardware on COM5!`
6. **Open the Dashboard:** Open your Google Chrome or Microsoft Edge browser and press `Ctrl + O` (Open File). Navigate to and open:
   `E:\sannidhi-main\sannidhi-main\dashboard\frontend\index.html`

The dashboard will initially say **"BUFFERING"** for 5 seconds (while it collects the first 500 packets to calculate the initial CSI-Bench normalization matrix). After 5 seconds, you will see live inference probabilities and real-time CSI Amplitude streaming to the chart!

---

## Architecture Note: CSI-Bench Compatibility
Your data processing is handled by `backend/csi_processor_paper.py`. This perfectly replicates the methodology from Section 4.2 of the CSI-Bench paper:
1. **Phase Elimination**: Converts raw I/Q into pure Amplitude.
2. **Segmentation**: Buffers data into 5-second (500 packet) temporal windows.
3. **Matrix Normalization**: Calculates global Mean ($\mu$) and Standard Deviation ($\sigma$) over the entire (500, 64) matrix to properly scale the amplitude, preventing domain shift when feeding data to your model.
