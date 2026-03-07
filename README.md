# AI Driver Monitoring System

## Project Overview

The **AI Driver Monitoring System** is a real-time computer vision application that detects unsafe driver behavior using deep learning and facial analysis. The system analyzes live video from an in-vehicle camera to identify fatigue, distraction, phone use, and loss of attention—helping prevent accidents and improve road safety.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Face Detection** | MediaPipe Face Mesh for robust facial landmark detection |
| **Eye Aspect Ratio (EAR)** | Blink and fatigue detection via geometric eye metrics |
| **PERCLOS Analysis** | Percentage of eye closure over time (fatigue indicator) |
| **Head Pose Estimation** | Yaw and pitch angles for gaze/attention direction |
| **Eye Gaze Direction** | Real-time estimation of where the driver is looking |
| **Phone Detection** | YOLOv8-based detection of phone use while driving |
| **Hand Detection** | Detects hands near the face (eating, grooming, etc.) |
| **Attention Analysis** | Combines gaze, head pose, and PERCLOS for attention score |
| **Adaptive Risk Engine** | Multi-factor risk scoring with configurable thresholds |
| **Real-Time Alerts** | Audio alerts (LOW / MEDIUM / HIGH) based on risk level |
| **Live Risk Graph** | Matplotlib-based trend visualization |
| **Trip Reports** | JSON export of session metrics and risk events |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Application                         │
│                    (Tkinter GUI + Video Loop)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
     ▼                       ▼                       ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│ Face Module │      │ Fatigue      │      │ Head Pose       │
│ (MediaPipe) │──────│ Module (EAR) │──────│ Module (yaw/    │
└─────────────┘      │ PERCLOS      │      │ pitch)          │
                     └──────────────┘      └─────────────────┘
     │                       │                       │
     │                       ▼                       │
     │               ┌──────────────┐                │
     │               │ Attention    │                │
     │               │ Analyzer     │                │
     │               └──────────────┘                │
     │                       │                       │
     ├───────────────────────┼───────────────────────┤
     ▼                       ▼                       ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│ Phone       │      │ Risk Engine  │      │ Gaze Estimator  │
│ (YOLOv8)    │      │ (Adaptive    │      │ + Hand Detector │
└─────────────┘      │  Scoring)    │      └─────────────────┘
                     └──────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │ Trip Report  │
                     │ (JSON)       │
                     └──────────────┘
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core language |
| **OpenCV** | Video capture, frame processing, image operations |
| **MediaPipe** | Face mesh, hand detection, landmarks |
| **YOLOv8 (Ultralytics)** | Phone object detection |
| **PyTorch** | Deep learning backend for YOLO |
| **NumPy** | Numerical operations |
| **SciPy** | Distance computations (EAR) |
| **Matplotlib** | Live risk trend graph |
| **Pillow (PIL)** | Image handling for Tkinter |
| **Tkinter** | Cross-platform GUI |

---

## Example Use Case

A fleet manager deploys the system in company vehicles. During a trip:

1. The driver starts a session via the **Open Camera** button.
2. The system processes each frame: detecting face, eyes, head pose, gaze, hands, and phone.
3. If the driver looks down at a phone, the risk score rises and a **HIGH** alert plays.
4. If the driver shows prolonged eye closure (PERCLOS), a **MEDIUM** fatigue alert triggers.
5. The live graph shows risk over time; the right panel displays driver state and risk level.
6. When the driver stops the session, a JSON trip report is saved for later analysis.

---

## Why This Project Matters for Road Safety

- **Drowsiness causes ~100,000 crashes/year** (NHTSA). EAR and PERCLOS can catch micro-sleeps before they lead to an accident.
- **Phone use increases crash risk by ~4x**. Real-time phone detection enables immediate alerts.
- **Distraction (eating, grooming, looking away)** is a leading cause of incidents. Head pose and hand detection help identify these behaviors.
- **Trip reports** support fleet safety programs, driver coaching, and compliance documentation.

---

## Project Structure

```
Driver-Monitoring-System/
├── main.py              # Entry point, GUI, video loop
├── config.py            # Thresholds and risk limits
├── modules/
│   ├── face_module.py   # Face detection
│   ├── fatigue_module.py# EAR, blink detection
│   ├── headpose_module.py # Head pose (yaw/pitch)
│   ├── gaze_module.py   # Gaze direction
│   ├── attention_module.py # PERCLOS, attention
│   ├── phone_module.py  # YOLOv8 phone detection
│   ├── hand_module.py   # Hand detection
│   ├── risk_module.py   # Risk scoring engine
│   └── trip_report.py   # JSON report generation
├── reports/             # Trip JSON outputs
└── sounds/              # Alert sounds (optional)
```

---

## Quick Start

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the application (requires webcam)
python main.py
```

**Note:** On first run, YOLOv8 will download the `yolov8s.pt` model if not present.

---

## License

This project is intended for educational and research purposes. Use responsibly in compliance with local regulations and privacy laws.
