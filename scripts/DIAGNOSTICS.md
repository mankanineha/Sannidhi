# Startup diagnostics

These scripts capture evidence for dashboard and CLI freezes without changing
the application code. Close Arduino Serial Monitor and any other program using
the ESP32 port before running them.

From the project root (`sannidhi-main`):

```cmd
scripts\run_diagnostics.bat dashboard COM5 60
scripts\run_diagnostics.bat cli COM5 60
```

Replace `COM5` with the port shown in Device Manager. The wrapper starts the
selected entry point, records stdout/stderr with timestamps, checks installed
packages and model files, probes the serial port, and writes a process
heartbeat every five seconds. It stops after 60 seconds, or sooner with
`Ctrl+C`.

Each run creates a folder under `diagnostics\`, containing:

- `run.log`: the main issue log, including the last message before a freeze.
- `preflight.json`: Python, package, model-file, and serial-port details.

Attach both files from the same run to the issue. Do not attach raw CSI CSVs
unless they are specifically requested; they may contain much more data than
needed.
