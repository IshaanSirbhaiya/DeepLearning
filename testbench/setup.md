# SafeEdge - Testbench Setup & Run Instructions

## Prerequisites

- Python 3.10+
- pip (Python package manager)
- Internet connection (for first-time dependency install)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `OPENAI_API_KEY` - For AI-generated incident summaries (optional - system falls back to templates)
- `TELEGRAM_BOT_TOKEN` - For Telegram notifications (optional for backend-only test)
- `GOOGLE_MAPS_API_KEY` - For outdoor evacuation routing (optional for backend-only test)

## Step 3: Run Backend Server

```bash
python -m backend.server
```

The server starts at `http://localhost:8000`. API documentation is auto-generated at `http://localhost:8000/docs`.

## Step 4: Run Detection on Sample Video

```bash
python -m detection.detector --input testbench/sample_fire.mp4
```

This runs YOLOv8n fire/smoke detection on the sample video. Detected alerts are sent to the backend API.

## Step 5: Run Full End-to-End Demo

```bash
python testbench/run_demo.py
```

This script:
1. Starts the backend server
2. Sends simulated fire detection alerts
3. Shows alerts appearing in the system
4. Generates AI incident summaries
5. Demonstrates the store-and-forward offline capability

## Expected Output

- Terminal shows fire detection results with bounding boxes and confidence scores
- Backend receives JSON alerts and stores them in SQLite
- AI summary generated for each alert (or template-based if no OpenAI key)
- Telegram bot sends evacuation messages to registered users (if configured)
- `alerts/` folder contains privacy-preserving snapshots (faces blurred)

## Test the API Manually

```bash
# Send a test alert
curl -X POST http://localhost:8000/alert \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "CAM_01",
    "event": "fire_detected",
    "confidence": 0.92,
    "risk_score": "CRITICAL",
    "location": {"building": "Block 4A", "floor": 3, "zone": "kitchen"}
  }'

# View all alerts
curl http://localhost:8000/alerts

# View stats
curl http://localhost:8000/stats
```
