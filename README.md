# PROJECT GARUDA: Multimodal AI for Motorcycle Rider Intent Prediction & Driver Monitoring System (DMS)

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)
![V2X](https://img.shields.io/badge/V2X-UDP%20Broadcast-0B7189)
![Status](https://img.shields.io/badge/Status-Production-16825D)

PROJECT GARUDA is a multimodal edge-AI system for motorcycle rider intent prediction and driver monitoring. It combines forward video, rider-facing visual features, and inertial telemetry to forecast rider intent, detect fatigue and distraction, surface safety alerts on a live HUD, broadcast machine-readable V2X warnings, and produce an auditable ride report.

> **Operational note:** The application is production-oriented, but any deployment on a moving vehicle must be validated against the target camera, lighting, helmet configuration, network, and local safety requirements before live operation.

## System Architecture

```text
                                  +-----------------------------+
                                  |  HUD Display (OpenCV)       |
                                  |  Intent / DMS / IMU status   |
                                  +--------------+--------------+
                                                 ^
                                                 |
+------------------+      +---------------------+---------------------+
| Camera            |----->| Dual Stream Feature Extraction            |
| Rider + road view |      | Vision: intent, face, hands, gear, gaze    |
+------------------+      | IMU: acceleration, gyro, lean, dynamics     |
                            +---------------------+---------------------+
                                                  |
                                                  v
                            +------------------------------------------+
                            | Cross-Attention Transformer              |
                            | T-2.5s rider-intent and status fusion    |
                            +-----------+---------------+--------------+
                                        |               |
                         +--------------+               +----------------+
                         v                                               v
              +----------------------+                         +----------------------+
              | Audio Siren          |                         | V2X UDP Broadcaster |
              | Microsleep wake-up   |                         | JSON safety beacon  |
              +----------------------+                         +----------------------+

                            +------------------------------------------+
                            | Automated PDF ReportLab Generator       |
                            | Flight log, incidents, metrics, score    |
                            +------------------------------------------+

  Hardware path: ESP32 + MPU6050 (I2C) ---------------------> IMU telemetry
```

## Feature Matrix

| Capability | Delivered behavior | Primary signals |
| --- | --- | --- |
| **T-2.5s Intent Forecasting** | Forecasts lane keep, left turn, right turn, and hard brake intent before the maneuver horizon. | Rider head pose, optical motion, speed change, multimodal model |
| **Real-time DMS** | Tracks Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), yawns, gaze, head pose, and drowsiness. Fires a microsleep alarm after 5 seconds of sustained eye closure. | Face landmarks, eye and mouth aperture, temporal state |
| **Rider Gear Compliance** | Reports specs, earphones, hands-on-bars state, and handheld-phone violations. | Face and hand regions, rider status classifier |
| **Autonomous telemetry HUD** | Displays intent confidence, rider state, speed, roll/pitch horizon, V2X status, FPS, and inference timing. | OpenCV HUD renderer |
| **V2X safety broadcast** | Sends an IPv4 UDP broadcast containing predicted intent, confidence, rider status, mitigation, and a safety command. | `255.255.255.255:37020` by default |
| **Automated PDF flight log** | Exports a timestamped PDF with session metrics, intent distribution, fatigue/compliance data, incident log, and an alertness-based safety score. | ReportLab, `reports/` |

## Project Layout

```text
main.py                         Live camera, inference loop, HUD, session controls
app.py                          Streamlit command dashboard and benchmark controls
api/server.py                   FastAPI health, trajectory prediction, and telemetry routes
pipeline/detector_tracker.py   YOLO pose detection and ByteTrack-compatible rider IDs
pipeline/feature_engine.py     Bounded per-rider kinematic and pose feature windows
pipeline/risk_engine.py        TTC, proximity, and green/yellow/red risk assessment
models/intent_forecaster.py    Five-class baseline/sequence forecast contract and SHAP hooks
utils/benchmark_eval.py        Offline metrics and synthetic SUMO/CARLA-style scenarios
models/multimodal_transformer.py Cross-attention multimodal model
dms/feature_extractor.py        Face, eye, mouth, hand, gaze, and compliance features
v2x/v2x_broadcast.py            UDP V2X safety beacon
utils/pdf_reporter.py            RideSession and ReportLab export
hardware/esp32_imu_hud.ino      ESP32 MPU6050 and RGB visor LED firmware
dataset/rider_dataset.py         Training dataset loader
train.py                         Model training entry point
generate_synthetic_data.py       Synchronized training-data generator
checkpoints/                     Saved model weights
data/                            Metadata, IMU clips, and video inputs
reports/                         Generated PDF ride reports
```

## Setup and Execution

### 1. Create the virtual environment

Windows PowerShell:

```powershell
python -m venv .venv-2
.\.venv-2\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv-2
source .venv-2/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Optional: generate data and train

The repository includes data and a model checkpoint. To regenerate synchronized synthetic data or retrain:

```bash
python generate_synthetic_data.py
python train.py --epochs 5
```

Training writes the best checkpoint to `checkpoints/best_intent_model.pth`.

### 4. Run the live application

```bash
python main.py
```

### Intent safety platform

Run the interactive dashboard:

```bash
streamlit run app.py
```

Run the inference API:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

The API accepts a bounded trajectory history at `POST /predict/trajectory`, returns
five-class intent probabilities, projected future coordinates, feature values, and
collision risk. `GET /health` and `POST /telemetry/log` are available for service
health and edge telemetry integration.

The default camera is index `0`. Useful runtime modes:

```bash
python main.py --camera 0                 # Select a camera index
python main.py --no-broadcast             # Disable UDP V2X messages
python main.py --test-mode                # Desktop/lab telemetry mode
python main.py --test-mode --simulate-motion
```

### Interactive controls

| Key | Action |
| --- | --- |
| `SPACE` / `S` | Start or stop the GARUDA session. Stopping exports a PDF report. |
| `Q` | Quit the application. |
| Arrow keys in `--test-mode` | Adjust simulated speed and lean telemetry. |

Generated reports use the form `reports/GARUDA_REPORT_YYYYMMDD_HHMMSS.pdf`.

## V2X Payload

When broadcasting is enabled, GARUDA sends JSON over UDP broadcast on port `37020`:

```json
{
  "timestamp": "2026-09-29T12:00:00+00:00",
  "predicted_intent": "INTENT_HARD_BRAKE",
  "confidence": 0.98,
  "predicted_status": "ALERT",
  "lookahead_sec": 2.5,
  "safety_mitigation": "DECEL_MAINTAIN_DISTANCE",
  "safety_command": "DRIVER_STATUS_ALERT"
}
```

Ensure the host firewall and local network permit UDP broadcast before connecting another vehicle or roadside receiver.

## Hardware Schematics

The reference firmware is [`hardware/esp32_imu_hud.ino`](hardware/esp32_imu_hud.ino). Use a common-ground, 3.3 V-compatible MPU6050 breakout and place a current-limiting resistor in series with each LED channel.

### ESP32 wiring table

| ESP32 pin | Connects to | Function / notes |
| --- | --- | --- |
| `3V3` | MPU6050 `VCC` | Sensor supply; verify the breakout voltage requirement. |
| `GND` | MPU6050 `GND` and LED common cathode | Shared reference ground. |
| `GPIO 21` | MPU6050 `SDA` | I2C data. |
| `GPIO 22` | MPU6050 `SCL` | I2C clock. |
| `GPIO 25` | Green LED anode through resistor | Clear / normal status. |
| `GPIO 26` | Yellow LED anode through resistor | Warning status. |
| `GPIO 27` | Red LED anode through resistor | Emergency status. |

The firmware uses MPU6050 address `0x68`, reads gyro and accelerometer channels at 100 Hz, and accepts serial LED commands at `115200` baud: `C` clear, `W` warning, and `E` emergency.

```text
ESP32 3V3  -------- MPU6050 VCC
ESP32 GND  -------- MPU6050 GND -------- LED common cathode
ESP32 GPIO21 ------ MPU6050 SDA
ESP32 GPIO22 ------ MPU6050 SCL
ESP32 GPIO25 --R--> Green LED anode
ESP32 GPIO26 --R--> Yellow LED anode
ESP32 GPIO27 --R--> Red LED anode
```

## Safety and Deployment Notes

- Treat predictions and compliance signals as driver-assistance telemetry, not as a replacement for rider judgment or certified vehicle controls.
- Test camera framing, night scenes, visor reflections, rain, occlusion, and helmet fit before field use.
- Keep UDP broadcast on a controlled network and validate the receiving system's command handling before connecting it to vehicle actuation.
- Keep generated reports and model checkpoints out of source control when they contain operational or personally identifiable data.

## Team / Submission

| Item | Details |
| --- | --- |
| Project | **PROJECT GARUDA** |
| Submission title | **Intelligent Rider Intent & Autonomous Telemetry HUD** |
| Team / organization | _Add team name_ |
| Contributors | _Add contributor names and roles_ |
| Repository / submission ID | _Add link or identifier_ |
| Demo artifact | Live HUD, ESP32 IMU telemetry, V2X beacon, and generated PDF report |
| Primary deliverables | Source code, trained checkpoint, hardware sketch, setup instructions, and safety report |

## License

Add the project license and third-party notices here before external distribution.