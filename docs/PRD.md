PRD: SafeEdge - Edge-Based Fire Safety Intelligence System
DLW 2026 | Track 3: AI in Security
Repo: https://github.com/IshaanSirbhaiya/DeepLearning
Deadline: March 3rd, 13:00 SGT
Team: 4 people (all vibe coding)

1. Problem
Communities face public safety challenges from fragmented infrastructure, slow emergency response, and outdated monitoring. When fire breaks out, detection relies on smoke detectors (which trigger late), and communication is manual - residents call 995, operators triage, responders are dispatched. This wastes critical minutes.

Real-world anchor: The October 2024 Singapore Singtel outage took down 995/999 emergency lines for hours, exposing centralized telecom as a single point of failure.

2. Solution
An edge-deployed fire detection system that:

Runs YOLOv8n on existing CCTV feeds locally to detect fire/smoke in real-time
Generates structured, risk-scored alerts with privacy-preserving snapshots
Sends evacuation instructions to residents via Telegram bot (indoor + outdoor routing)
Provides dynamic Google Maps evacuation routes that avoid the fire zone
Optionally displays alerts on a web dashboard for authorities (stretch goal)
3. System Architecture

┌─────────────────────────────────────────────────────┐
│                  DETECTION LAYER                     │
│              (You + Teammate - 2 people)             │
│                                                      │
│  [CCTV Feed / Webcam / Video File]                  │
│         │                                            │
│         ▼                                            │
│  [YOLOv8n Fire/Smoke Detection]                     │
│         │                                            │
│         ▼                                            │
│  [Risk Scoring Engine]                               │
│    - Confidence threshold filtering                  │
│    - Multi-frame confirmation (reduce false pos)     │
│         │                                            │
│         ▼                                            │
│  [Privacy Filter] ── blur faces in snapshot          │
│         │                                            │
│         ▼                                            │
│  [Alert Generator]                                   │
│    Output: JSON alert + blurred snapshot             │
│    {                                                 │
│      "camera_id": "CAM_01",                         │
│      "event": "fire_detected",                      │
│      "confidence": 0.92,                            │
│      "risk_score": "CRITICAL",                      │
│      "timestamp": "2026-03-02T14:23:01",            │
│      "location": {"building": "Block 4A",           │
│                    "floor": 3, "zone": "kitchen"},   │
│      "snapshot_path": "alerts/snap_001_blurred.jpg" │
│    }                                                 │
│         │                                            │
└─────────┼────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│              COMMUNICATION LAYER                     │
│            (2 Teammates - 2 people)                  │
│                                                      │
│  [Alert Receiver] ◄── receives JSON from detection  │
│         │                                            │
│         ├──► [Telegram Bot]                          │
│         │      - Sends fire alert to registered      │
│         │        users (building/area based)         │
│         │      - Includes:                           │
│         │        • Fire location description         │
│         │        • Indoor evacuation directions      │
│         │        • Google Maps evacuation link       │
│         │        • Blurred snapshot                  │
│         │                                            │
│         ├──► [Indoor Evacuation Engine]              │
│         │      - Pre-mapped building graph           │
│         │        (nodes = rooms/stairwells/exits)    │
│         │      - Dijkstra/A* pathfinding             │
│         │      - Avoids fire-affected zones          │
│         │      - Output: text directions             │
│         │        "Exit via Stairwell B (North).      │
│         │         Avoid Floor 3 corridor."           │
│         │                                            │
│         └──► [Outdoor Evacuation Engine]             │
│                - Google Maps Directions API          │
│                - Dynamic routing AROUND fire zone    │
│                - Sends Maps link to assembly point   │
│                                                      │
│  [STRETCH: Web Dashboard for Authorities]            │
│    - Real-time alert feed                            │
│    - Building map with fire locations                │
│    - Camera status overview                          │
│                                                      │
└─────────────────────────────────────────────────────┘
4. Component Specifications
4.1 Fire Detection Engine (YOUR SCOPE)
Input: Video feed (CCTV/webcam/mp4 file)

Model: YOLOv8n (nano) - pre-trained fire/smoke weights

Use fastest available: search Kaggle/HuggingFace for "YOLOv8 fire smoke detection" weights
If no good pre-trained exists: fine-tune YOLOv8n on a fire dataset from Kaggle (e.g., "fire-smoke-dataset")
Target: runs at 15+ FPS on laptop CPU (proving edge viability)
Detection Pipeline:

Ingest video frame-by-frame via OpenCV
Run YOLOv8n inference on each frame
Filter detections: confidence > 0.5
Multi-frame confirmation: require fire detected in N consecutive frames (e.g., 5 out of 8) to avoid false positives from flickering lights etc.
When confirmed: capture snapshot, apply face blur (OpenCV + MediaPipe/Haar cascades), generate JSON alert
Pass alert to Communication layer (via API call, file write, or message queue)
Output: JSON alert object + blurred snapshot image (see schema in architecture above)

Edge Metrics to Log:

FPS (inference speed)
CPU/memory usage (via psutil)
Model size on disk
Purpose: prove in docs/video that this runs on constrained hardware
4.2 Risk Scoring
Simple but effective:

confidence < 0.5 → ignore (noise)
0.5 - 0.7 → WARNING (possible fire, monitor)
0.7 - 0.9 → HIGH (likely fire, alert sent)
> 0.9 → CRITICAL (confirmed fire, immediate alert)
Multi-frame confirmation adds reliability - log the frame count to show judges the system doesn't cry wolf.

4.3 Privacy Filter
Detect faces in snapshot using OpenCV Haar cascades or MediaPipe Face Detection
Apply Gaussian blur to face bounding boxes
Save blurred snapshot only - raw frame never leaves the "edge node"
This directly addresses the "privacy-preserving" requirement from the track
4.4 Indoor Evacuation (TEAMMATES' SCOPE)
Pre-map ONE sample building as a weighted graph:
Nodes: rooms, corridors, stairwells, exits
Edges: paths between nodes with distance weights
Fire-affected nodes/edges get infinite weight (blocked)
Run Dijkstra or A* from user's location to nearest safe exit
Output: text directions ("Go to Stairwell B, descend to Level 1, exit North Gate")
4.5 Outdoor Evacuation (TEAMMATES' SCOPE)
Google Maps Directions API
Origin: building exit point
Destination: nearest pre-defined assembly point
Dynamic avoidance: route around the fire zone
Output: Google Maps URL link sent via Telegram
4.6 Telegram Bot (TEAMMATES' SCOPE)
Users register with bot: /register <building> <floor>
Bot stores user preferences (building, floor, Telegram chat ID)
On fire alert:
Filter users in affected building
Send message with: fire info + indoor directions + Google Maps link + snapshot
Demo: team members register their own phones
4.7 Web Dashboard (STRETCH GOAL)
Simple real-time page showing:
Alert feed (most recent first)
Map with camera/building locations
Alert status (active/resolved)
Tech: HTML/JS + WebSocket or polling from backend
Only build if core system is done and working
5. Tech Stack
Component	Technology	Reason
Fire detection	Python + Ultralytics YOLOv8n	Lightweight, nano model proves edge viability
Video processing	OpenCV	Standard, reliable, fast
Face blur	OpenCV Haar / MediaPipe	Lightweight privacy filter
Backend/API	FastAPI or Flask	Quick to set up, connects detection → communication
Telegram bot	python-telegram-bot library	Mature, well-documented
Indoor pathfinding	NetworkX (Python)	Graph algorithms built-in (Dijkstra, A*)
Outdoor routing	Google Maps Directions API	Dynamic routing with avoidance
AI summarization	OpenAI API (GPT-4o-mini)	Summarize alerts for authorities (use $100 credits)
Database	SQLite	Lightweight, zero-config, stores alerts + user registrations
Edge metrics	psutil	Log CPU/memory/FPS
Dashboard (stretch)	HTML/JS + Chart.js	Keep it simple
6. Data Flow (Happy Path)

1. CCTV feed streams frames to detection engine
2. YOLOv8n detects fire/smoke (confidence 0.87)
3. Multi-frame confirmation: 6/8 frames positive → CONFIRMED
4. Risk score: HIGH
5. Snapshot captured → faces blurred → saved locally
6. JSON alert generated → sent to backend API
7. Backend receives alert:
   a. Stores in SQLite
   b. Queries registered Telegram users in "Block 4A"
   c. Runs indoor pathfinding: "Floor 3 kitchen fire → avoid east corridor → Stairwell B"
   d. Generates Google Maps link: building exit → assembly point (avoiding fire zone)
   e. OpenAI summarizes: "Fire detected Floor 3 Kitchen, Block 4A. High confidence. 2 cameras confirm."
   f. Telegram bot sends to all registered Block 4A users:
      "🔥 FIRE ALERT - Block 4A, Floor 3
       Confidence: 87% | Risk: HIGH

       EVACUATE NOW:
       → Head to Stairwell B (North)
       → Avoid Floor 3 East Corridor
       → Exit via North Gate

       Assembly Point: [Google Maps Link]

       [Blurred snapshot attached]"
8. (Stretch) Dashboard updates with new alert
7. Repository Structure

DeepLearning/
├── README.md                    # Setup, dependencies, how to run
├── requirements.txt             # All Python dependencies
├── .env.example                 # Template for API keys (Telegram, Google Maps, OpenAI)
├── .gitignore                   # .env, __pycache__, etc.
│
├── detection/                   # DETECTION LAYER (Your scope)
│   ├── detector.py              # Main fire detection engine
│   ├── risk_scorer.py           # Risk scoring logic
│   ├── privacy_filter.py        # Face blur module
│   ├── alert_generator.py       # JSON alert creation
│   └── models/                  # YOLOv8n weights
│       └── fire_smoke.pt
│
├── communication/               # COMMUNICATION LAYER (Teammates' scope)
│   ├── telegram_bot.py          # Telegram bot logic
│   ├── indoor_evacuation.py     # Graph + pathfinding
│   ├── outdoor_routing.py       # Google Maps integration
│   ├── building_graphs/         # Pre-mapped building data
│   │   └── block_4a.json        # Sample building graph
│   └── assembly_points.json     # Pre-defined safe assembly locations
│
├── backend/                     # BACKEND (connects everything)
│   ├── server.py                # FastAPI/Flask server
│   ├── database.py              # SQLite operations
│   └── summarizer.py            # OpenAI alert summarization
│
├── dashboard/                   # STRETCH: Web dashboard
│   └── index.html               # Simple HTML/JS dashboard
│
├── testbench/                   # REQUIRED: Judges test from here
│   ├── setup.md                 # Step-by-step instructions
│   ├── sample_fire.mp4          # Sample fire video
│   ├── sample_smoke.mp4         # Sample smoke video
│   ├── sample_no_fire.mp4       # Negative test case
│   └── run_demo.py              # One-click demo script
│
├── docs/                        # Documentation
│   └── SafeEdge_Documentation.pdf
│
└── alerts/                      # Generated alert snapshots
    └── .gitkeep
8. Work Division
Detection Team (You + Teammate 1)
Task	Description
Find/train YOLO model	Search for pre-trained YOLOv8 fire weights OR fine-tune on Kaggle fire dataset
Build detection pipeline	OpenCV video ingestion → YOLO inference → multi-frame confirmation
Risk scoring	Implement confidence thresholds + multi-frame logic
Privacy filter	Face detection + Gaussian blur on snapshots
Alert generator	Create JSON alerts with all required metadata
API endpoint	Expose detection results via FastAPI endpoint for communication layer to consume
Edge metrics	Log FPS, CPU, memory to prove edge viability
Testbench samples	Download/source sample fire/smoke videos for testbench/
Communication Team (Teammates 2 + 3)
Task	Description
Telegram bot	User registration, alert broadcasting, message formatting
Indoor evacuation	Map sample building as graph, implement Dijkstra/A*, generate text directions
Outdoor routing	Google Maps Directions API integration, dynamic fire zone avoidance
Alert receiver	Consume JSON alerts from detection API
Assembly points	Define sample safe assembly locations
Building graph	Create the pre-mapped building JSON for demo
Integration + Docs (Split across team)
Task	Owner
Connect detection → backend → Telegram	All 4 (integration session)
README.md	You (detection team lead)
testbench/setup.md	You
PDF Documentation (IEEE citations)	Assign 1 person
2-minute video	Assign 1 person
OpenAI summarization	Whoever finishes first
Dashboard (stretch)	Whoever finishes first
9. Timeline
Day 1 (Today - March 1)
Detection team: Find YOLO fire model, build detection pipeline, get it working on a sample video
Communication team: Set up Telegram bot, start building indoor graph + pathfinding
End of day goal: Detection engine outputs JSON alerts from a fire video. Telegram bot can send messages.
Day 2 (March 2)
Morning: Finish individual components. Detection team: add risk scoring + privacy filter. Communication team: finish outdoor routing + Google Maps.
Afternoon: INTEGRATION SESSION - connect detection → backend API → Telegram bot. End-to-end flow working.
Evening: testbench/ folder, README.md, start PDF documentation. If time: dashboard.
Day 3 (March 3 - DEADLINE 13:00)
Morning (before 11:00 AM): Final testing, record 2-min demo video, finish PDF, make repo public
11:00 AM - 13:00: Submit on portal, verify everything uploaded correctly
10. Testbench Strategy (CRITICAL - judges grade from this)
testbench/setup.md should contain:


# SafeEdge - Test Instructions

## Prerequisites
- Python 3.10+
- Telegram Bot Token (create via @BotFather)
- Google Maps API Key
- OpenAI API Key

## Setup
1. Clone: `git clone https://github.com/IshaanSirbhaiya/DeepLearning.git`
2. Install: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in API keys
4. Download YOLO weights: `python download_model.py` (or include in repo if small enough)

## Run Detection Only
python detection/detector.py --input testbench/sample_fire.mp4

## Run Full System (Detection + Telegram + Evacuation)
python testbench/run_demo.py

## Expected Output
- Terminal: JSON alerts printed as fire is detected
- Telegram: Bot sends alert messages to registered users
- alerts/ folder: Blurred snapshots saved
Include 2-3 sample videos in testbench/ (keep small, <20MB each).

11. Judging Criteria Mapping
Criterion	How SafeEdge scores
Empathy & Impact	Solves real fire safety gaps. Privacy-first (face blur, local processing, no cloud video). Telegram delivery is accessible to all residents.
Feasibility & Scalability	YOLOv8n is ~6MB, runs on CPU at 15+ FPS. Uses existing CCTV cameras. Telegram works on low bandwidth. Add cameras via config.
Innovation	Multi-signal fire confirmation (CV + multi-frame). Dynamic evacuation routing that avoids fire zones. Edge-first architecture. Privacy-preserving by design.
Technical Implementation	Real inference pipeline, graph-based pathfinding, API integration (Google Maps, Telegram, OpenAI), proper alert schema.
Presentation & Documentation	Clean testbench/, IEEE-cited PDF, 2-min demo showing full flow from fire detection → Telegram alert → evacuation.
12. Demo Video Script (2 minutes)
Problem (15s): "When fire breaks out, smoke detectors are slow, and 995 operators are overwhelmed. We built SafeEdge."
Detection demo (30s): Show sample fire video → YOLO detecting fire → JSON alert generated → blurred snapshot
Telegram alert (30s): Show phone receiving Telegram message with fire info + indoor directions + Google Maps link
Evacuation path (20s): Show the indoor graph visualization + outdoor Google Maps route avoiding fire
Edge metrics (10s): Flash the FPS/CPU/memory stats proving lightweight operation
Closing (15s): "SafeEdge: upgrades existing CCTV with edge AI. No new hardware. Privacy-preserving. Works on low bandwidth."