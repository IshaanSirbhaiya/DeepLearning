# SafeEdge - Edge-Based Fire Safety Intelligence System

**DLW 2026 | Track 3: AI in Security | NTU Singapore**

SafeEdge upgrades existing CCTV cameras with lightweight, on-device AI to detect fires **before they spread** - requiring zero new hardware, zero infrastructure changes, and zero cloud dependency. When fire or smoke is detected, the system generates risk-scored alerts, confirms with AI vision, triggers evacuation routing, and notifies residents via Telegram - all within seconds.

> **Why SafeEdge?** Singapore experiences 3,000+ fire incidents annually (SCDF). Traditional smoke detectors take 30-90 seconds to trigger. The 2024 Singtel outage took down 995/999 emergency lines for hours. SafeEdge detects fire in under 1 second, works fully offline, and extends the evacuation window by 30-120 seconds.

## The Problem We Solve

When a fire breaks out today, a smoke detector eventually triggers an alarm - but someone must **manually investigate** to confirm whether it's real. This verification step takes 2-5 minutes, during which the fire grows and escape routes become compromised. Only after manual confirmation do residents get notified, and evacuation is uncoordinated - people panic, stampede, and don't know the safest route out.

**SafeEdge eliminates this bottleneck entirely.** Our AI detects fire visually, confirms it with a second AI opinion, and triggers coordinated evacuation - all within 5 seconds. By the time a human would have manually reached the scene to check, SafeEdge has already confirmed the fire, notified every resident via Telegram with the optimal evacuation route, and the command dashboard is tracking who has reached safety and who still needs rescue. No new hardware required - it runs on existing CCTV cameras.

Our team discovered this problem by analyzing Singapore's emergency response data on Data.gov.sg. Fire emerged as one of the most frequent and preventable hazards, with over 3,000 incidents annually. The October 2024 Singtel outage - which took down 995/999 emergency lines - confirmed that any solution must work offline, without depending on centralized telecom infrastructure.

> **Full documentation:** See [`docs/SafeEdge_Documentation.pdf`](docs/SafeEdge_Documentation.pdf) for methodology, design trade-offs, testing procedures, observations, and IEEE citations.

## Key Features

- **Dual-track fire detection** - YOLOv8n (ML) + EarlyFireDetector (optical flow + heat shimmer + haze analysis)
- **Pre-fire early warning** - detects heat shimmer and pixel distortion 30+ seconds before visible flames
- **2-Factor AI confirmation** - OpenAI GPT-4o-mini Vision provides second-opinion on every detection
- **Multi-frame confirmation** - 5/8 sliding window eliminates false positives from flickering lights
- **Privacy-preserving** - faces auto-blurred via MediaPipe, raw video never leaves the edge node
- **Edge-first architecture** - ~6MB model, 15+ FPS on laptop CPU, no cloud dependency
- **Sentinel-Mesh dashboard** - real-time evacuation command centre with NTU campus map
- **SOS tracking** - tracks who reached safe zones, who's unaccounted, who needs rescue
- **Telegram alerts** - instant fire notifications with indoor/outdoor evacuation routing
- **Google Maps routing** - dynamic outdoor evacuation avoiding fire zones
- **Sensor-fusion ready** - architecture designed to integrate IoT smoke/temperature sensor data
- **Intelligence reports** - 3 PDF reports with real data charts + AI-generated narratives

## Architecture

```
[Existing CCTV / Webcam]
    |
    +---> [YOLOv8n Fire/Smoke Detection]     (Track A: ML-based)
    +---> [EarlyFireDetector]                 (Track B: optical flow + haze)
    |
    v
[Risk Scorer] --> [Multi-frame 5/8 confirmation]
    |
    v
[Privacy Filter] --> [Face blur via MediaPipe]
    |
    v
[Alert Generator] --> [OpenAI Vision 2FA confirmation]
    |
    v
[Fire Event Bus] --> /fire endpoint (port 8001)
    |
    +---> [Sentinel-Mesh Dashboard]  (Streamlit + Supabase)
    +---> [Telegram Bot]             (evacuation alerts)
    +---> [Google Maps]              (outdoor routing)
    +---> [PDF Reports]              (intelligence analytics)
```

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API Key (for Vision 2FA + report narratives)

### Setup

```bash
git clone https://github.com/IshaanSirbhaiya/DeepLearning.git
cd DeepLearning
pip install -r requirements.txt
# Create .env with your API keys
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Run Detection (webcam)

```bash
cd detection
python detector.py --input 0 --server
# Opens webcam + starts FastAPI server on port 8001
```

### Run Detection (video file)

```bash
cd detection
python detector.py --input ../testbench/sample_fire.mp4
```

### Run Dashboard

```bash
streamlit run app.py
# Opens Sentinel-Mesh Fire Command Centre
```

### Generate Intelligence Reports

```bash
python -m reports.generate_reports          # All 3 reports with AI narratives
python -m reports.generate_reports --no-ai  # Quick mode without OpenAI
```

### Run Synthetic Demo

```bash
cd detection
python demo.py
# Shows synthetic fire scene with HUD overlay
```

## Project Structure

```
DeepLearning/
+-- detection/                   # Fire Detection Layer
|   +-- detector.py              # Main YOLOv8n engine + FastAPI server
|   +-- early_detector.py        # Pre-fire anomaly detection (optical flow)
|   +-- risk_scorer.py           # Multi-frame risk scoring
|   +-- privacy_filter.py        # Face blur (MediaPipe)
|   +-- alert_generator.py       # JSON alerts + OpenAI Vision 2FA
|   +-- fire_event.py            # Fire event bus + /fire API
|   +-- demo.py                  # Synthetic fire demo
|   +-- models/                  # YOLOv8n weights (~6MB)
|   +-- alerts/                  # Generated alert snapshots + JSON
|
+-- app.py                       # Sentinel-Mesh Streamlit dashboard
+-- campus.graphml               # NTU walking network (offline pathfinding)
+-- supabase_schema.sql          # Database schema
|
+-- reports/                     # PDF Intelligence Reports
|   +-- generate_reports.py      # Entry point
|   +-- report_fire_trends.py    # Report 1: SCDF fire statistics
|   +-- report_emergency_response.py  # Report 2: SCDF response analysis
|   +-- report_system_performance.py  # Report 3: Real alert metrics
|
+-- docs/                        # Documentation + Generated PDFs
|   +-- SafeEdge_Documentation.pdf
|   +-- SafeEdge_Fire_Trends.pdf
|   +-- SafeEdge_Emergency_Response.pdf
|   +-- SafeEdge_System_Performance.pdf
|
+-- testbench/                   # Test materials for judges
+-- requirements.txt
+-- .env
```

## API Endpoints (port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/fire` | Latest fire event (polling) |
| GET | `/fire/history?n=10` | Last N fire events |
| POST | `/fire/clear` | Clear fire state |
| GET | `/health` | Service health check |
| GET | `/status` | Detector status (FPS, CPU, memory) |
| GET | `/alerts/latest` | Latest alert details |
| GET | `/early-warning` | Pre-fire anomaly state |

## Documentation

See [`docs/SafeEdge_Documentation.pdf`](docs/SafeEdge_Documentation.pdf) for the full technical documentation with design trade-offs, architecture details, performance results, and IEEE citations.

## Team

Built for Deep Learning Week 2026 Hackathon | MLDA@EEE | NTU Singapore
