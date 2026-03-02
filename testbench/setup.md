# SafeEdge — Testbench Setup & Run Instructions

**For DLW 2026 Judges | Track 3: AI in Security**

SafeEdge upgrades existing CCTV cameras with lightweight, on-device AI to detect fires before they spread — requiring zero new hardware, zero infrastructure changes, and zero cloud dependency.

---

## Quick Start (Single Command)

```bash
pip install -r requirements.txt
python testbench/run_demo.py
```

**That's it.** No API keys needed. This runs the entire pipeline in one terminal:

| Phase | What happens |
|-------|-------------|
| **1. Detection** | YOLO + EarlyFireDetector on video — CV2 window with live bounding boxes |
| **2. Communication** | Simulated Telegram alerts + evacuation routing with real math |
| **3. Dashboard** | Streamlit Command Centre opens in browser with fire map + evacuee status |

Press **Q** in the video window to skip ahead. Press **Ctrl+C** to exit.

### Options

```bash
python testbench/run_demo.py --input path/to/video.mp4   # custom video
python testbench/run_demo.py --no-display                 # headless (no CV2 window)
python testbench/run_demo.py --skip-dashboard             # skip Streamlit launch
```

---

## Prerequisites

- Python 3.10+
- pip (Python package manager)
- Internet connection (first run only — downloads ~6MB YOLO model)

## What the Demo Shows

### Phase 1: Fire Detection (~30s)

- **EarlyFireDetector** flags pre-fire anomalies (optical flow + background subtraction + texture variance) BEFORE flames appear
- **YOLOv8n** detects fire/smoke with bounding boxes and confidence scores
- Multi-frame confirmation: 5/8 consecutive frames required to trigger alert
- Terminal prints real-time detection events and structured alert JSON

### Phase 2: Communication & Routing (~5s)

- Simulates Telegram broadcast to 4 registered users
- Calculates real Haversine distances from each user to the fire
- Routes endangered users to the nearest of 9 NTU assembly zones
- Generates Google Maps walking directions
- Simulates user button taps: "I have reached Safety" / "EMERGENCY RESCUE"
- Writes local state to `testbench/demo_state.json`

### Phase 3: Command Centre Dashboard

- Streamlit dashboard opens at http://localhost:8501
- NTU campus map (Folium) with fire hazard zone and assembly markers
- KPI cards: Notified / Safe / Unaccounted / Active SOS
- Reads from `demo_state.json` — no Supabase needed

---

## Optional: Run Individual Subsystems

For deeper testing, each component can run independently.

### Live Webcam Detection

```bash
cd detection
python detector.py --input 0 --server
```

Opens webcam with real-time YOLO detection. Hold your phone showing a fire video to test.

### API Endpoints (with --server flag)

```bash
curl http://localhost:8001/health         # service health
curl http://localhost:8001/status         # FPS, CPU, memory
curl http://localhost:8001/fire           # latest fire event
curl http://localhost:8001/early-warning  # pre-fire anomalies
```

### Dashboard Only

```bash
streamlit run app.py
```

Works with mock data if no Supabase credentials are configured.

### Intelligence Reports

```bash
python -m reports.generate_reports --no-ai    # no API key needed
python -m reports.generate_reports             # with OpenAI narratives
```

Generates 3 PDF reports: Fire Trends, Emergency Response, System Performance.

---

## API Keys (Optional)

All features work without API keys. To enable cloud services, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Key | Purpose | Without it |
|-----|---------|-----------|
| `OPENAI_API_KEY` | Vision 2FA (GPT-4o confirms fire) | 2FA skipped, YOLO-only detection |
| `TELEGRAM_BOT_TOKEN` | Real Telegram alerts | Alerts simulated in terminal |
| `SUPABASE_URL` + `SUPABASE_KEY` | Live dashboard sync | Dashboard uses local state file |

---

## Troubleshooting

- **No bounding boxes?** Ensure video contains visible fire/smoke. Model requires conf > 0.45.
- **Model download on first run?** YOLOv8n weights (~6MB) auto-download from HuggingFace.
- **`ModuleNotFoundError`?** Run `pip install -r requirements.txt` from the project root.
- **Port 8501 in use?** Kill existing Streamlit: `streamlit kill` or use a different port.

---

*SafeEdge — Edge-Based Fire Safety Intelligence System | DLW 2026 | Track 3: AI in Security | NTU Singapore*
