# SafeEdge - Edge-Based Fire Safety Intelligence System

**DLW 2026 | Track 3: AI in Security | NTU Singapore**

SafeEdge upgrades existing CCTV cameras with lightweight, on-device AI to detect fires **before they spread** - requiring zero new hardware, zero infrastructure changes, and zero cloud dependency. When fire or smoke is detected, the system generates risk-scored alerts, confirms with AI vision, triggers evacuation routing, and notifies residents via Telegram - all within seconds.

> **Why SafeEdge?** Singapore experiences 3,000+ fire incidents annually (SCDF). Traditional smoke detectors take 30-90 seconds to trigger. The 2024 Singtel outage took down 995/999 emergency lines for hours. SafeEdge detects fire in under 1 second, works fully offline, and extends the evacuation window by 30-120 seconds.

---

## For Judges: Quick Test (Single Command)

**No API keys required.** Run from the project root:

```bash
pip install -r requirements.txt
python testbench/run_demo.py
```

This runs the FULL pipeline in one terminal:

| Phase | What happens |
|-------|-------------|
| **1. Detection** | YOLO + EarlyFireDetector on video with live bounding boxes |
| **2. Communication** | Interactive evacuation — you play as a trapped person |
| **3. Dashboard** | Streamlit Command Centre opens in browser |

See [`testbench/setup.md`](testbench/setup.md) for detailed instructions and options.

---

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
- **Telegram alerts** - instant fire notifications with indoor/outdoor evacuation routing via inline buttons
- **Google Maps routing** - dynamic walking directions to nearest of 9 NTU assembly zones
- **Intelligence reports** - 3 PDF reports with real data charts + AI-generated narratives

## Benchmark & Simulation Results

| Test | Type | F1 | Precision | Recall | Key Finding |
|------|------|----|-----------|--------|-------------|
| **FireSense Benchmark** | Real video (100 clips) | 92.4% | 85.9% | 100% | 22x faster alerts (14s vs 309s), 34% evac reduction |
| **1000-Scenario Simulation** | Synthetic Monte Carlo | 92.7% | 99.4% | 86.9% | Only 3 FPs across 1000 scenarios |
| **NTU Campus Simulation** | Building-specific | - | - | - | Occupancy modeling + network failure resilience |

> **FireSense benchmark** used 100 real fire/non-fire video clips from a citable dataset (DOI: [10.5281/zenodo.836749](https://doi.org/10.5281/zenodo.836749)). Full report: [`SafeEdge_Report2_FireSense.pdf`](SafeEdge_Report2_FireSense.pdf)

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
    |                   Saves to detection/alerts/ (JSON + blurred JPG)
    v
[Fire Event Bus] --> /fire endpoint (port 8001)
    |
    +---> [Sentinel-Mesh Dashboard]  (Streamlit + Supabase)
    +---> [Telegram Bot]             (evacuation alerts + inline buttons)
    +---> [Google Maps]              (walking route to assembly zones)
    +---> [PDF Reports]              (intelligence analytics)
    +---> [Simulation Framework]     (1000-scenario + NTU campus)
```

## Quick Start

### Prerequisites
- Python 3.10+
- No API keys required for core detection and testbench demo

### Setup

```bash
git clone https://github.com/IshaanSirbhaiya/DeepLearning.git
cd DeepLearning
pip install -r requirements.txt
```

### Optional: API Keys

Copy `.env.example` to `.env` to enable cloud services:

| Key | Purpose | Without it |
|-----|---------|-----------|
| `OPENAI_API_KEY` | Vision 2FA (GPT-4o confirms fire) | 2FA skipped, YOLO-only detection |
| `TELEGRAM_BOT_TOKEN` | Real Telegram alerts to users | Alerts simulated in terminal |
| `SUPABASE_URL` + `SUPABASE_KEY` | Live dashboard sync | Dashboard uses local state file |

### Run Full Demo (single command)

```bash
python testbench/run_demo.py
```

### Run Detection (webcam)

```bash
cd detection
python detector.py --input 0 --serve
# Opens webcam + starts FastAPI server on port 8001
```

### Run Telegram Bot (evacuation routing)

```bash
python mesh_router.py
# Polls /fire endpoint, sends alerts, routes users to safe zones
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

### Run Simulations

```bash
python safeedge_simulation.py --no-ai       # 1000-scenario benchmark
python safeedge_simulation2.py --no-ai      # NTU campus simulation
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
|   +-- supabase_publisher.py    # Pushes fire events to Supabase
|   +-- demo.py                  # Real video + live YOLO detection demo
|   +-- models/                  # YOLOv8n weights (~6MB, auto-downloaded)
|   +-- alerts/                  # Real detection snapshots (JSON + blurred JPG)
|
+-- app.py                       # Sentinel-Mesh Streamlit dashboard
+-- mesh_router.py               # Telegram bot + NetworkX evacuation routing
+-- campus.graphml               # NTU walking network (offline pathfinding)
+-- supabase_schema.sql          # Database schema
+-- .env.example                 # Environment variable template
|
+-- reports/                     # PDF Report Generation
|   +-- generate_reports.py      # Entry point (3 intelligence reports)
|   +-- doc_generator.py         # SafeEdge_Documentation.pdf generator
|   +-- report_fire_trends.py    # Report 1: SCDF fire statistics
|   +-- report_emergency_response.py  # Report 2: SCDF response analysis
|   +-- report_system_performance.py  # Report 3: Real alert metrics
|
+-- docs/                        # Generated Documentation
|   +-- SafeEdge_Documentation.pdf
|   +-- PRD.md
|
+-- testbench/                   # Test materials for judges
|   +-- setup.md                 # Step-by-step setup & run instructions
|   +-- run_demo.py              # Single-command E2E demo (zero API keys)
|   +-- sample_fire.mp4          # Test video
|
+-- safeedge_simulation.py       # 1000-scenario Monte Carlo simulation
+-- safeedge_simulation2.py      # NTU campus-specific simulation
+-- SafeEdge_Simulation_Report.pdf        # Simulation 1 results
+-- SafeEdge_Campus_Simulation_Report.pdf # Simulation 2 results
+-- SafeEdge_Report2_FireSense.pdf        # Real video benchmark (100 clips)
+-- SafeEdge_Simulation_Report_stats.json # Simulation metrics (JSON)
+-- requirements.txt
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
