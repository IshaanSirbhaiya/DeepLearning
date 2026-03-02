PRD: SafeEdge - Edge-Based Fire Safety Intelligence System
DLW 2026 | Track 3: AI in Security
Repo: https://github.com/IshaanSirbhaiya/DeepLearning
Deadline: March 3rd, 13:00 SGT
Team: 4 people

1. Problem
Communities face public safety challenges from fragmented infrastructure, slow emergency response, and outdated monitoring. When fire breaks out, detection relies on smoke detectors (which trigger late), and communication is manual - residents call 995, operators triage, responders are dispatched. This wastes critical minutes.

Real-world anchor: The October 2024 Singapore Singtel outage took down 995/999 emergency lines for hours, exposing centralized telecom as a single point of failure.

2. Solution
An edge-deployed fire detection system that:

- Runs YOLOv8n on existing CCTV feeds locally to detect fire/smoke in real-time
- Detects pre-fire anomalies (heat shimmer, haze) via optical flow + background subtraction before visible flames
- Confirms detections with OpenAI Vision API (GPT-4o-mini) as a second-opinion check
- Generates structured, risk-scored alerts with privacy-preserving snapshots
- Publishes fire events to a real-time evacuation dashboard (Sentinel-Mesh)
- Tracks evacuee status and SOS signals via Supabase backend
- Provides dynamic NTU campus evacuation with 9 pre-defined safe assembly zones
- Generates PDF intelligence reports with charts and AI-powered narratives

3. System Architecture

┌─────────────────────────────────────────────────────┐
│                  DETECTION LAYER                     │
│          (Integration branch - 2 people)             │
│                                                      │
│  [CCTV Feed / Webcam / Video File]                  │
│         │                                            │
│         ├──► [YOLOv8n Fire/Smoke Detection]          │
│         │     Track A: ML-based (fire, smoke, flame) │
│         │                                            │
│         └──► [EarlyFireDetector]                     │
│               Track B: CPU-only pre-fire warnings    │
│               (optical flow + bg sub + texture)      │
│         │                                            │
│         ▼                                            │
│  [Risk Scoring Engine]                               │
│    - Confidence threshold filtering                  │
│    - Multi-frame confirmation (5 of 8 frames)        │
│         │                                            │
│         ▼                                            │
│  [Privacy Filter] ── blur faces (MediaPipe)          │
│         │                                            │
│         ▼                                            │
│  [Alert Generator]                                   │
│    - JSON alert + blurred snapshot                   │
│    - OpenAI Vision 2FA (GPT-4o-mini confirms)        │
│    Output: {                                         │
│      "camera_id": "CAM_01",                         │
│      "event": "fire_and_smoke",                     │
│      "confidence": 0.85,                            │
│      "risk_score": "HIGH",                          │
│      "location": {"building": "The Hive",           │
│                    "floor": 2, "zone": "Studio"},    │
│      "vision_analysis": { ... GPT-4o-mini ... }     │
│    }                                                 │
│         │                                            │
│  [Fire Event Bus] ── /fire endpoint (port 8001)     │
│         │                                            │
└─────────┼────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│         COMMUNICATION & DASHBOARD LAYER              │
│        (ntu-hackathon branch - 2 people)             │
│                                                      │
│  [Sentinel-Mesh: Fire Command Centre]               │
│    - Streamlit real-time dashboard                   │
│    - Folium map with NTU campus overlay             │
│    - 9 safe assembly zones (green markers)          │
│    - Fire hazard zone (red radius circle)           │
│    - 4 KPI cards: Notified / Safe / Unaccounted /   │
│      Active SOS                                      │
│    - Mesh Telemetry Feed (live event log)           │
│                                                      │
│  [Supabase Backend]                                  │
│    - users table: id, status, lat, lon              │
│    - fire_locations table: incident, lat, lon,      │
│      radius_m, is_active                            │
│    - Real-time polling for dashboard updates        │
│                                                      │
│  [NTU Campus Graph]                                  │
│    - OSMnx walking network (44K nodes)              │
│    - Offline pathfinding capability                  │
│                                                      │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│              INTELLIGENCE REPORTS                    │
│                                                      │
│  [3 PDF Reports with Charts + AI Narratives]        │
│    1. Fire Incident Trend Analysis (SCDF data)      │
│    2. Emergency Response Optimization (SCDF data)   │
│    3. System Performance Report (real alert data)   │
│                                                      │
│  Generated via: python -m reports.generate_reports  │
│  Uses: matplotlib charts + OpenAI GPT-4o-mini text  │
│                                                      │
└─────────────────────────────────────────────────────┘

4. Component Specifications
4.1 Fire Detection Engine
Input: Video feed (CCTV/webcam/mp4 file)

Model: YOLOv8n (nano) - pre-trained fire/smoke weights from HuggingFace (keremberke/yolov8n-fire-smoke-detection)
- ~6MB model, runs at 15+ FPS on laptop CPU
- Classes: fire, smoke, flame
- Confidence threshold: 0.45

Detection Pipeline (Dual-Track):
- Track A (YOLOv8n): ML-based fire/smoke detection per frame
- Track B (EarlyFireDetector): CPU-only pre-fire anomaly detection using:
  - Optical flow magnitude (motion anomalies)
  - Background subtraction (scene changes)
  - Texture variance analysis (shimmer/haze)
  - Combined threshold: 0.55, requires 2 of 3 signals, 6 confirmation frames

Multi-frame confirmation: require fire detected in 5 out of 8 consecutive frames
When confirmed: capture snapshot, apply face blur, generate JSON alert, call OpenAI Vision

4.2 OpenAI Vision 2FA (Second-Opinion Confirmation)
After YOLO confirms fire, the blurred snapshot is sent to GPT-4o-mini Vision API.
Returns structured analysis: fire_visible, smoke_visible, risk_level, false_positive_likely, description.
This provides AI-powered false positive reduction beyond multi-frame confirmation.

4.3 Risk Scoring
confidence < 0.5 → ignore (noise)
0.5 - 0.7 → WARNING (possible fire, monitor)
0.7 - 0.9 → HIGH (likely fire, alert sent)
> 0.9 → CRITICAL (confirmed fire, immediate alert)

4.4 Privacy Filter
Detect faces in snapshot using MediaPipe Face Detection
Apply Gaussian blur to face bounding boxes
Raw frame never leaves the edge node — only blurred snapshots are saved/transmitted

4.5 NTU Campus Locations
4 hardcoded campus locations with lat/lng coordinates:
- The Hive (Floor 2, Collaboration Studio)
- Northspine (Floor 1, Food Court)
- School of Chemical and Biomedical Engineering (Floor 3, Laboratory Wing)
- Hall of Residence 2 (Floor 4, Common Kitchen)

4.6 Sentinel-Mesh Dashboard (Communication Layer)
Streamlit-based Fire Command Centre with:
- Live Folium map showing fire epicenter + safe assembly zones
- 9 NTU safe assembly zones (North Spine Plaza, South Spine Plaza, SRC, etc.)
- Real-time user status tracking (secure/safe/unaccounted/SOS)
- Supabase backend for persistent state
- Mock data fallback for demo without Supabase credentials
- Auto-sync every 3 seconds

4.7 Intelligence Reports
3 PDF reports generated with matplotlib charts and OpenAI GPT-4o-mini narratives:
1. Fire Incident Trend Analysis — Singapore SCDF fire statistics (2014-2024)
2. Emergency Response Optimization — SCDF division response times and patterns
3. System Performance Report — Real alert data (confidence, FPS, CPU, memory)

5. Tech Stack
Component          | Technology                    | Reason
Fire detection     | Python + Ultralytics YOLOv8n  | Lightweight, ~6MB proves edge viability
Pre-fire detection | OpenCV (optical flow, bg sub) | CPU-only, no ML model needed
Video processing   | OpenCV                        | Standard, reliable, fast
Face blur          | MediaPipe Face Detection      | Lightweight privacy filter
AI confirmation    | OpenAI GPT-4o-mini Vision     | Second-opinion false positive reduction
Detection API      | FastAPI (port 8001)           | Fire event bus + health endpoints
Dashboard          | Streamlit + Folium            | Real-time evacuation command centre
Database           | Supabase (PostgreSQL)         | Cloud-hosted, real-time subscriptions
Campus graph       | NetworkX + OSMnx             | Offline pathfinding on NTU walking network
Reports            | fpdf2 + matplotlib            | PDF generation with charts
AI narratives      | OpenAI GPT-4o-mini            | Professional report text generation
Edge metrics       | psutil                        | Log CPU/memory/FPS

6. Data Flow (Happy Path)

1. CCTV feed streams frames to detection engine
2. Dual-track detection:
   a. YOLOv8n detects fire/smoke (confidence 0.85)
   b. EarlyFireDetector flags heat shimmer anomaly (30s before visible flames)
3. Multi-frame confirmation: 5/8 frames positive → CONFIRMED
4. Risk score: HIGH
5. Snapshot captured → faces blurred → saved locally
6. OpenAI Vision 2FA confirms: "fire_visible: true, false_positive_likely: false"
7. JSON alert generated → published to Fire Event Bus
8. Fire Event Bus:
   a. /fire endpoint returns latest event (teammates poll this)
   b. Supabase publisher inserts into fire_locations table
9. Sentinel-Mesh dashboard updates:
   a. Map shows fire hazard zone at detected location
   b. KPI cards update (Notified, Safe, Unaccounted, SOS)
   c. Telemetry feed shows real-time status changes
10. Intelligence reports can be regenerated with latest alert data

7. Repository Structure

DeepLearning/
├── README.md
├── requirements.txt
├── .env                         # API keys (OPENAI_API_KEY, SUPABASE_URL, etc.)
├── .gitignore
│
├── detection/                   # DETECTION LAYER (integration branch)
│   ├── detector.py              # Main YOLOv8n fire detection engine + FastAPI
│   ├── early_detector.py        # Pre-fire anomaly detection (optical flow + bg sub)
│   ├── risk_scorer.py           # Multi-frame risk scoring logic
│   ├── privacy_filter.py        # Face blur module (MediaPipe)
│   ├── alert_generator.py       # JSON alert creation + OpenAI Vision 2FA
│   ├── fire_event.py            # Fire event bus + /fire API endpoint
│   ├── demo.py                  # Synthetic fire scene demo with HUD overlay
│   ├── models/
│   │   └── fire_smoke.pt        # YOLOv8n weights (auto-downloaded from HuggingFace)
│   └── alerts/                  # Real detection snapshots (JSON + blurred JPG pairs)
│
├── app.py                       # Sentinel-Mesh Streamlit dashboard (ntu-hackathon)
├── campus.graphml               # NTU walking network graph (OSMnx)
├── supabase_schema.sql          # Database schema for Supabase
├── seed_mock_data.py            # Mock data seeder for dashboard demo
├── fetch_map.py                 # Script to download NTU campus graph
│
├── backend/                     # Alert aggregation & persistence
│   ├── server.py                # FastAPI server (port 8000)
│   ├── database.py              # SQLite persistence
│   └── summarizer.py            # AI summary generation
│
├── reports/                     # PDF report generation
│   ├── generate_reports.py      # Entry point (--report 1|2|3, --no-ai)
│   ├── doc_generator.py         # SafeEdge_Documentation.pdf generator
│   ├── report_fire_trends.py    # Report 1: SCDF fire statistics
│   ├── report_emergency_response.py  # Report 2: SCDF response analysis
│   ├── report_system_performance.py  # Report 3: Real alert data analytics
│   ├── pdf_theme.py             # SafeEdgePDF branded template
│   ├── ai_narrator.py           # OpenAI GPT-4o-mini narrative generator
│   └── data_fetcher.py          # Data.gov.sg fetcher + SCDF fallback data
│
├── testbench/                   # Test materials for judges
│   ├── setup.md                 # Step-by-step setup & run instructions
│   ├── run_demo.py              # Single-command E2E demo (zero API keys)
│   └── sample_fire.mp4
│
├── testing_procedures/          # Benchmark Suite (FireSense dataset)
│   ├── run_suite.py             # Auto-downloads FireSense, slices clips, runs E2E
│   └── USAGE.txt                # How to run the benchmark suite
│
├── docs/                        # Generated documentation + PDFs
│   ├── PRD.md
│   ├── SafeEdge_Documentation.pdf
│   ├── SafeEdge_Fire_Trends.pdf
│   ├── SafeEdge_Emergency_Response.pdf
│   └── SafeEdge_System_Performance.pdf
│
├── safeedge_simulation.py       # 1000-scenario Monte Carlo simulation
├── safeedge_simulation2.py      # NTU campus-specific simulation
├── SafeEdge_Simulation_Report.pdf        # Simulation 1 results
├── SafeEdge_Campus_Simulation_Report.pdf # Simulation 2 results
├── SafeEdge_Report2_FireSense.pdf        # Real video benchmark (100 clips)
└── SafeEdge_Simulation_Report_stats.json # Simulation metrics (JSON)

8. API Endpoints

Detection Layer (port 8001):
  GET  /fire              — Latest fire event (polling endpoint for teammates)
  GET  /fire/history?n=10 — Last N fire events
  POST /fire/clear        — Clear fire state after incident resolved
  GET  /health            — Service health check
  GET  /status            — Detector status (FPS, CPU %, memory)
  GET  /alerts/latest     — Latest alert details
  GET  /alerts/drain      — Drain alert queue
  GET  /early-warning     — Early warning state (pre-fire anomalies)

9. How to Run

# Detection (webcam):
cd detection && python detector.py --input 0 --server

# Detection (video file):
cd detection && python detector.py --input ../testbench/sample_fire.mp4

# Synthetic demo:
cd detection && python demo.py

# Dashboard (Streamlit):
streamlit run app.py

# Generate PDF reports:
python -m reports.generate_reports          # All 3 reports with AI
python -m reports.generate_reports --no-ai  # Without OpenAI (faster)
python -m reports.generate_reports --report 3  # System performance only

10. Judging Criteria Mapping
Criterion | How SafeEdge scores
Empathy & Impact | Solves real fire safety gaps (Singtel 995 outage). Privacy-first (face blur, local processing). Edge-first = no cloud dependency.
Feasibility & Scalability | YOLOv8n is ~6MB, runs on CPU at 15+ FPS. Uses existing CCTV. Supabase scales horizontally. Add cameras via config.
Innovation | Dual-track detection (ML + optical flow). Pre-fire anomaly detection (30s early warning). OpenAI Vision 2FA. Edge-first architecture. Privacy-preserving by design.
Technical Implementation | Real inference pipeline, multi-frame confirmation, OpenAI Vision integration, Streamlit dashboard, Supabase real-time backend, PDF reports with charts.
Presentation & Documentation | 3 PDF intelligence reports, clean testbench, 2-min demo showing full detection → dashboard flow.

11. Benchmark & Simulation Testing

11.1 FireSense Real-Video Benchmark (Report 2)
Tested SafeEdge's full YOLO pipeline on 100 real fire/non-fire clips from the FireSense dataset (DOI: 10.5281/zenodo.836749).
Results: 100% recall, 85.9% precision, F1 = 92.4%. Alerts generated 22x faster than traditional detection (14s vs 309s). 34% evacuation time reduction.
Report: SafeEdge_Report2_FireSense.pdf (project root)

11.2 1000-Scenario Simulation (Simulation 1)
Monte Carlo framework generating 1000 synthetic scenarios across 12 categories (large/small/night fires, smoke, heat shimmer, haze, cooking steam, exhaust, reflections, sunlight, crowds, empty corridors).
Results: 99.4% precision, 86.9% recall, F1 = 92.7%. Only 3 false positives. 43.8% evacuation time reduction.
Run: python safeedge_simulation.py --no-ai
Report: SafeEdge_Simulation_Report.pdf + SafeEdge_Simulation_Report_stats.json (project root)

11.3 NTU Campus Simulation (Simulation 2)
Campus-specific simulation with 15 NTU buildings, occupancy levels, multi-camera coordination, and mesh network failure resilience.
Run: python safeedge_simulation2.py --no-ai
Report: SafeEdge_Campus_Simulation_Report.pdf (project root)

12. Demo Video Script (2 minutes)
Problem (15s): "When fire breaks out, smoke detectors are slow, and the 2024 Singtel outage proved 995 can fail. We built SafeEdge."
Detection demo (30s): Show fire video → YOLO detecting fire/smoke → JSON alert with OpenAI Vision confirmation → blurred snapshot
Dashboard (30s): Show Sentinel-Mesh dashboard updating with fire location, safe zones, evacuee status
Early detection (20s): Show EarlyFireDetector catching heat shimmer before visible flames
Edge metrics (10s): Flash FPS/CPU/memory stats proving lightweight edge operation
Reports (10s): Show 3 PDF intelligence reports with real charts and AI narratives
Closing (5s): "SafeEdge: edge AI fire detection for Singapore. No new hardware. Privacy-preserving. Works offline."
