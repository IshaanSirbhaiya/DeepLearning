# SafeEdge — Testbench Setup & Run Instructions

**For DLW 2026 Judges | Track 3: AI in Security**

SafeEdge upgrades existing CCTV cameras with lightweight, on-device AI to detect fires before they spread — requiring zero new hardware, zero infrastructure changes, and zero cloud dependency.

---

## Prerequisites

- Python 3.10+
- pip (Python package manager)
- Webcam (optional — can use provided test video instead)
- Internet connection (for first-time dependency install and OpenAI Vision 2FA)

## Step 1: Install Dependencies

```bash
cd DeepLearning
pip install -r requirements.txt
```

## Step 2: Configure API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_key_here
```

- `OPENAI_API_KEY` — Required for Vision 2FA (second-opinion fire confirmation) and AI-generated report narratives. Detection still works without it, but Vision 2FA will be skipped.

Optional keys (for dashboard and notifications):
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
TELEGRAM_BOT_TOKEN=your_telegram_token
GOOGLE_MAPS_API_KEY=your_google_maps_key
```

## Step 3: Run Fire Detection

### Option A — Webcam (live detection)

```bash
cd detection
python detector.py --input 0 --server
```

This opens your webcam feed with real-time YOLOv8n fire/smoke detection and starts a FastAPI server on port 8001.

**To test:** Hold your phone up to the webcam showing a fire video from YouTube (search "fire burning footage"). You should see bounding boxes appear around fire/smoke regions with confidence scores.

### Option B — Test video (no webcam needed)

```bash
cd detection
python detector.py --input ../testbench/sample_fire.mp4
```

This runs detection on the provided sample video.

### Option C — Synthetic demo (no video needed)

```bash
cd detection
python demo.py
```

This generates a synthetic fire scene with the full HUD overlay showing detection metrics.

## Step 4: Verify Detection Output

When fire is detected, check:

1. **Terminal output** — Shows detection events with confidence scores and risk levels
2. **HUD overlay** — Bounding boxes around fire/smoke with labels (fire, smoke, flame)
3. **Alert JSON** — Generated in `detection/alerts/` folder:
   ```
   detection/alerts/
   ├── alert_YYYYMMDD_HHMMSS.json    # Structured alert with risk score
   └── alert_YYYYMMDD_HHMMSS.jpg     # Privacy-preserved snapshot (faces blurred)
   ```
4. **Vision 2FA** — If `OPENAI_API_KEY` is set, the alert JSON will contain a `vision_analysis` field:
   ```json
   {
     "vision_analysis": {
       "fire_visible": true,
       "smoke_visible": true,
       "risk_level": "HIGH",
       "false_positive_likely": false,
       "description": "Active flames visible..."
     }
   }
   ```
   If `vision_analysis` is `null`, the OpenAI API key is not configured.

## Step 5: Test the API Endpoints

With the detection server running (`--server` flag), test the API at `http://localhost:8001`:

```bash
# Check service health
curl http://localhost:8001/health

# Get detector status (FPS, CPU, memory)
curl http://localhost:8001/status

# Get latest fire event (teammates poll this endpoint)
curl http://localhost:8001/fire

# Get fire event history
curl http://localhost:8001/fire/history?n=10

# Get latest alert details
curl http://localhost:8001/alerts/latest

# Get early warning state (pre-fire anomalies)
curl http://localhost:8001/early-warning
```

## Step 6: Run Sentinel-Mesh Dashboard

```bash
streamlit run app.py
```

This opens the **Sentinel-Mesh Fire Command Centre** in your browser with:
- Live Folium map showing NTU campus with fire hazard zones
- 9 safe assembly zones (green markers)
- KPI cards: Notified / Safe / Unaccounted / Active SOS
- Mesh Telemetry Feed (real-time event log)
- Evacuee status tracking

The dashboard polls the detection server's `/fire` endpoint and updates Supabase in real-time. Works with mock data if Supabase credentials are not configured.

## Step 7: Generate Intelligence Reports

```bash
# Generate all 3 PDF reports with AI-powered narratives
python -m reports.generate_reports

# Generate without OpenAI (faster, uses template text)
python -m reports.generate_reports --no-ai

# Generate a specific report only
python -m reports.generate_reports --report 1   # Fire Trends
python -m reports.generate_reports --report 2   # Emergency Response
python -m reports.generate_reports --report 3   # System Performance
```

Reports are saved to `docs/`:
- `SafeEdge_Fire_Trends.pdf` — Singapore SCDF fire statistics (2014-2024)
- `SafeEdge_Emergency_Response.pdf` — SCDF response time analysis by division
- `SafeEdge_System_Performance.pdf` — Real detection metrics (confidence, FPS, CPU, memory)

## Expected Output Summary

| Component | What You Should See |
|-----------|-------------------|
| Detection | Bounding boxes around fire/smoke, confidence scores in terminal |
| Early Warning | Pre-fire anomaly flags before visible flames appear |
| Privacy Filter | Faces automatically blurred in saved snapshots |
| Vision 2FA | `vision_analysis` populated in alert JSON (with API key) |
| Alert Files | JSON + blurred JPG saved in `detection/alerts/` |
| API Server | Health, status, fire events available at port 8001 |
| Dashboard | NTU campus map with fire zones, safe zones, evacuee tracking |
| Reports | 3 branded PDF reports with charts and data-driven narratives |

## Troubleshooting

- **No bounding boxes appearing?** Ensure the video contains visible fire/smoke. The model requires confidence > 0.45 and 5/8 multi-frame confirmation before alerting.
- **`vision_analysis: null`?** Check that `OPENAI_API_KEY` is set in `.env` and the key is valid.
- **Model download on first run?** YOLOv8n weights (~6MB) are auto-downloaded from HuggingFace on first execution.
- **Dashboard shows no fire events?** Ensure the detection server is running with `--server` flag on port 8001.

---

*SafeEdge — Edge-Based Fire Safety Intelligence System | DLW 2026 | Track 3: AI in Security | NTU Singapore*
