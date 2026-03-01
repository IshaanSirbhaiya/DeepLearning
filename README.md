# SafeEdge - Edge-Based Fire Safety Intelligence System

**DLW 2026 | Track 3: AI in Security**

SafeEdge upgrades existing low-resolution CCTV cameras with lightweight, on-device AI to detect fires before they spread. When fire or smoke is detected, the system generates risk-scored alerts, sends evacuation instructions to residents via Telegram, and provides dynamic evacuation routing that avoids fire zones.

## Architecture

```
[Existing CCTV Camera]
    → [Edge Detection (YOLOv8n fire/smoke)]
    → [Risk Scoring + Privacy Filter (face blur)]
    → [FastAPI Backend (alert storage + AI summary)]
    → [Telegram Bot (resident alerts + evacuation paths)]
    → [Google Maps (outdoor routing avoiding fire zone)]
```

## Key Features

- **Early fire detection** using YOLOv8n on existing CCTV feeds
- **Multi-frame confirmation** to reduce false positives
- **Privacy-preserving** - faces auto-blurred, no raw video stored
- **Edge-first** - processes locally, sends only lightweight JSON alerts
- **Indoor evacuation** - graph-based pathfinding around fire zones
- **Outdoor routing** - dynamic Google Maps links avoiding danger areas
- **Telegram alerts** - instant notifications with evacuation directions
- **AI summaries** - OpenAI-generated incident reports for first responders
- **Store-and-forward** - works offline, syncs alerts when connection restored

## Quick Start

### Prerequisites
- Python 3.10+
- Telegram Bot Token (create via [@BotFather](https://t.me/botfather))
- OpenAI API Key (optional, for AI summaries)
- Google Maps API Key (for outdoor routing)

### Setup

```bash
git clone https://github.com/IshaanSirbhaiya/DeepLearning.git
cd DeepLearning
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run Backend Server

```bash
python -m backend.server
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Run Detection (on CCTV feed or video file)

```bash
python -m detection.detector --input testbench/sample_fire.mp4
```

### Run Full Demo

```bash
python testbench/run_demo.py
```

## Project Structure

```
DeepLearning/
├── backend/                     # API server + database + AI summarizer
│   ├── server.py                # FastAPI endpoints
│   ├── database.py              # SQLite alert storage
│   └── summarizer.py            # OpenAI incident summarization
├── detection/                   # Fire detection engine
│   ├── detector.py              # YOLOv8n inference pipeline
│   ├── risk_scorer.py           # Multi-frame confirmation + scoring
│   ├── privacy_filter.py        # Face blur module
│   ├── alert_generator.py       # JSON alert creation
│   └── models/                  # YOLO model weights
├── communication/               # Telegram bot + evacuation routing
│   ├── telegram_bot.py          # Resident notification bot
│   ├── indoor_evacuation.py     # Graph-based pathfinding
│   ├── outdoor_routing.py       # Google Maps integration
│   └── building_graphs/         # Pre-mapped building data
├── testbench/                   # Judge testing materials
│   ├── setup.md                 # Step-by-step test instructions
│   └── run_demo.py              # One-click demo script
├── alerts/                      # Generated alert snapshots
├── docs/                        # PDF documentation
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/alert` | Receive fire detection alert (JSON) |
| POST | `/alert/with-snapshot` | Receive alert with image snapshot |
| GET | `/alerts` | List recent alerts |
| GET | `/alerts/{id}` | Get single alert |
| GET | `/alerts/pending/forward` | Get unforwarded alerts (for Telegram bot) |
| POST | `/alerts/{id}/forwarded` | Mark alert as forwarded |
| POST | `/cameras` | Register a camera |
| GET | `/cameras` | List all cameras |
| GET | `/stats` | Alert statistics |

## Team

- **Detection Layer**: Fire detection engine (YOLOv8n + risk scoring + privacy filter)
- **Communication Layer**: Telegram bot + evacuation routing + Google Maps
- **Backend + Integration**: FastAPI server + database + OpenAI summarizer + testbench

## License

Built for Deep Learning Week 2026 Hackathon.
