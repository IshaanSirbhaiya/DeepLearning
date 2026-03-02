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
| **2. Communication** | Interactive evacuation — you play as a trapped person near the fire |
| **3. Dashboard** | Streamlit Command Centre opens in browser with fire map + evacuee status |

Press **Q** in the video window to skip ahead. Press **Ctrl+C** to exit.

### Options

```bash
python testbench/run_demo.py --input path/to/video.mp4   # custom video
python testbench/run_demo.py --no-display                 # headless (no CV2 window)
python testbench/run_demo.py --skip-dashboard             # skip Streamlit launch
python testbench/run_demo.py --no-interactive              # auto-simulate, no prompts
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

### Phase 2: Interactive Evacuation (~30s)

- You play as a trapped person near the fire
- Share your location → system calculates distance and routes you to nearest safe zone
- Choose: "I have reached Safety" or "EMERGENCY RESCUE (SOS)"
- Other registered users resolve automatically in the background
- Writes live state to `testbench/demo_state.json` (dashboard updates in real-time)

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

## Advanced: Full Setup with Real API Keys

To run the complete system with live Telegram alerts, OpenAI Vision 2FA, and Supabase dashboard sync, follow these steps.

### 1. OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com) → **API Keys** → **Create new secret key**
2. Add to your `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. This enables the Vision 2FA layer — GPT-4o-mini analyzes fire snapshots for a second opinion

### 2. Telegram Bot Setup

1. Open Telegram, search for **@BotFather**
2. Send `/newbot`, follow the prompts, and copy the bot token
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your-bot-token-here
   ```
4. **Get your Telegram User ID**: message **@userinfobot** on Telegram — it replies with your numeric user ID (e.g., `8112299954`)
5. Open `mesh_router.py` and add your user ID to the `REGISTERED_USERS` list (line 26):
   ```python
   REGISTERED_USERS = ["5808527465", "1674734061", "5574113481", "8112299954", "YOUR_ID_HERE"]
   ```
6. Start a chat with your bot on Telegram and send `/start`

### 3. Supabase Setup

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** → paste the contents of `supabase_schema.sql` → **Run**
3. Go to **Settings** → **API** → copy **Project URL** and **anon public key**
4. Add to `.env`:
   ```
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_KEY=eyJ...
   ```

### 4. Running the Full System (3 Terminals)

```bash
# Terminal 1: Fire Detection (webcam + FastAPI server)
cd detection && python detector.py --input 0 --server

# Terminal 2: Telegram Bot + Evacuation Routing
python mesh_router.py

# Terminal 3: Sentinel-Mesh Dashboard
streamlit run app.py
```

Hold your phone showing a fire video in front of the webcam. The system will:
1. Detect fire via YOLO → confirm with multi-frame scoring → send to OpenAI Vision 2FA
2. Publish alert → Telegram bot sends evacuation directions to all registered users
3. Dashboard updates in real-time with fire location, KPI cards, and evacuee status

---

## Simulation Testing

Run the 1000-scenario simulation framework (no API keys needed):

```bash
python safeedge_simulation.py --no-ai
```

This generates `SafeEdge_Simulation_Report_stats.json` and a PDF report in `docs/`. For the campus-wide multi-building simulation:

```bash
python safeedge_simulation2.py
```

---

## Troubleshooting

- **No bounding boxes?** Ensure video contains visible fire/smoke. Model requires conf > 0.45.
- **Model download on first run?** YOLOv8n weights (~6MB) auto-download from HuggingFace.
- **`ModuleNotFoundError`?** Run `pip install -r requirements.txt` from the project root.
- **Port 8501 in use?** Kill existing Streamlit: `streamlit kill` or use a different port.

---

*SafeEdge — Edge-Based Fire Safety Intelligence System | DLW 2026 | Track 3: AI in Security | NTU Singapore*
