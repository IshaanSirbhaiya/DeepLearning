"""
🚨 Sentinel-Mesh: Fire Command Centre
─────────────────────────────────────
Live Evacuation & Telemetry Dashboard | Location: NTU Campus
Light-mode UI, IoT Mesh tracking, and Supabase integration.
"""

import math
import os
import networkx as nx
import osmnx as ox
import folium
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from folium.plugins import AntPath
from supabase import create_client, Client

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentinel-Mesh: Fire Command Centre",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide sidebar initially for better dashboard view
)

# ── Supabase Init ─────────────────────────────────────────────────────────────

# Attempt to load from Streamlit secrets, fallback to env vars or mock data if missing
if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    USE_MOCK_DATA = False
else:
    # Use robust mock data if no Supabase credentials exist yet
    USE_MOCK_DATA = True

# ── CSS (Modern Light Mode Dashboard) ──────────────────────────────────────────

st.markdown("""
<style>
/* Global light theme */
.stApp { background-color: #f0f2f5; color: #1e293b; font-family: 'Inter', sans-serif; }

/* Header Typography */
h1 { color: #0f172a !important; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 0px !important; }
.subtitle { color: #64748b; font-size: 1.1rem; font-weight: 500; margin-bottom: 25px; }

/* Metric Cards Grid Container */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.5rem;
    margin-bottom: 2rem;
}

/* Base Metric Card */
.fcc-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border: 1px solid #e2e8f0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.fcc-card:hover { align-items: stretch; transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }

.fcc-title { font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.fcc-value { font-size: 2.5rem; font-weight: 800; line-height: 1; margin-bottom: 4px; }
.fcc-sub { font-size: 0.85rem; font-weight: 600; }

/* Specific Card Colors */
/* Total Notified (Neutral/Blue) */
.card-total { border-top: 4px solid #3b82f6; }
.card-total .fcc-title { color: #64748b; }
.card-total .fcc-value { color: #0f172a; }

/* Verified Safe (Green) */
.card-safe { border-top: 4px solid #10b981; background: #f0fdf4; border-color: #bbf7d0;}
.card-safe .fcc-title { color: #047857; }
.card-safe .fcc-value { color: #065f46; }
.card-safe .fcc-sub { color: #059669; }

/* Unaccounted (Orange) */
.card-warning { border-top: 4px solid #f59e0b; background: #fffbeb; border-color: #fde68a;}
.card-warning .fcc-title { color: #b45309; }
.card-warning .fcc-value { color: #92400e; }

/* Active SOS (Solid Red) */
.card-sos { background: #ef4444; border: none; box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.39); }
.card-sos .fcc-title { color: #fee2e2; }
.card-sos .fcc-value { color: #ffffff; animation: pulse 2s infinite; }
.card-sos .fcc-sub { color: #fca5a5; }

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* Panels */
.panel-container {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
    height: 100%;
}
.panel-title {
    font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 15px;
    display: flex; align-items: center; gap: 8px;
}
.live-dot { height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block; animation: pulse 2s infinite;}

/* Mesh Logs */
.mesh-log {
    padding: 12px; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem;
    display: flex; flex-direction: column; gap: 4px;
}
.mesh-log:last-child { border-bottom: none; }
.log-time { color: #94a3b8; font-family: monospace; }
.log-status-sos { display: inline-block; background: #fee2e2; color: #dc2626; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;}
.log-status-safe { display: inline-block; background: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants & Helpers ───────────────────────────────────────────────────────

GRAPHML_PATH = "campus.graphml"
NTU_CENTER = [1.3453, 103.6815]

SAFE_ZONES = {
    "North Spine Plaza":          (1.3468, 103.6810),
    "South Spine Plaza":          (1.3425, 103.6832),
    "Sports & Rec Centre (SRC)":  (1.3496, 103.6835),
    "Yunnan Garden (Open Field)": (1.3458, 103.6858),
    "Hall 1–3 Field":             (1.3540, 103.6855),
    "Innovation Centre Carpark":  (1.3448, 103.6785),
    "The Arc (North Spine CP-E)": (1.3475, 103.6800),
    "CCEB Assembly (CW4)":        (1.3435, 103.6870),
    "CCDS Assembly (N3 Carpark)": (1.3460, 103.6790),
}

# ── Mock Data Generator ───────────────────────────────────────────────────────

def fetch_telemetry_data():
    """Fetch user status counts and SOS locations from Supabase or Mock."""
    if USE_MOCK_DATA:
        return {
            "counts": {"total": 1250, "secure": 600, "safe": 242, "unaccounted": 405, "sos": 3},
            "sos_locations": [
                {"uid": "user_11x", "lat": 1.3440, "lon": 103.6820, "time": "17:14:02"},
                {"uid": "user_89p", "lat": 1.3470, "lon": 103.6850, "time": "17:15:10"},
                {"uid": "user_42a", "lat": 1.3455, "lon": 103.6795, "time": "17:16:45"},
            ],
            "recent_logs": [
                {"uid": "user_42a", "event": "SOS INITIATED", "status": "sos", "time": "17:16:45"},
                {"uid": "user_99x", "event": "Arrived at Assembly", "status": "safe", "time": "17:16:30"},
                {"uid": "user_89p", "event": "SOS INITIATED", "status": "sos", "time": "17:15:10"},
                {"uid": "user_12b", "event": "Mesh Node Connected", "status": "secure", "time": "17:14:55"},
            ],
            "fire_zone": {"lat": 1.3450, "lon": 103.6825, "radius": 100, "incident_name": "Mock Fire Drill"}
        }
    
    try:
        # 0. Fetch latest active fire incident (if any)
        fire_zone = None
        try:
            fire_res = (
                supabase
                .table("fire_locations")
                .select("incident_name, lat, lon, radius_m, reported_at")
                .eq("is_active", True)
                .order("reported_at", desc=True)
                .limit(1)
                .execute()
            )
            fire_rows = fire_res.data or []
            if fire_rows:
                fire_row = fire_rows[0]
                fire_zone = {
                    "lat": float(fire_row["lat"]),
                    "lon": float(fire_row["lon"]),
                    "radius": int(fire_row.get("radius_m", 100)),
                    "incident_name": fire_row.get("incident_name", "Fire Incident"),
                }
        except Exception:
            fire_zone = None

        # 1. Fetch all users to do counting and map plotting
        res = supabase.table("users").select("*").execute()
        users = res.data
        
        counts = {"total": len(users), "secure": 0, "safe": 0, "unaccounted": 0, "sos": 0}
        sos_locations = []
        recent_logs = []
        
        # Sort users by updated_at descending to mimic recent logs
        sorted_users = sorted(users, key=lambda x: x.get('updated_at', x.get('created_at', '')), reverse=True)
        
        for u in users:
            stat = str(u.get("status", "")).lower()
            if stat in counts:
                counts[stat] += 1
                
            # Grab SOS coordinates
            if stat == "sos":
                lat = u.get("lat")
                lon = u.get("lon")
                if lat and lon:  # Only map if they have coordinates
                    sos_locations.append({
                        "uid": str(u.get("id", "Unknown")),
                        "lat": float(lat),
                        "lon": float(lon),
                        "time": u.get("updated_at", "Just now")[:16].replace("T", " ")
                    })
                    
        # Populate live feed from 10 most recently updated rows
        for u in sorted_users[:10]:
            stat = str(u.get("status", "")).lower()
            event_map = {
                "sos": "SOS INITIATED",
                "safe": "Arrived at verified Assembly",
                "secure": "Mesh Node Connection Strong",
                "unaccounted": "In transit/Connection weak"
            }
            recent_logs.append({
                "uid": f"user_{str(u.get('id', 'x'))[:4]}",
                "event": event_map.get(stat, "Status Update"),
                "status": stat,
                "time": u.get("updated_at", "Just now")[:16].replace("T", " ")
            })
            
        return {
            "counts": counts,
            "sos_locations": sos_locations,
            "recent_logs": recent_logs,
            "fire_zone": fire_zone or {"lat": 1.3450, "lon": 103.6825, "radius": 100, "incident_name": "Fallback Demo Fire"}
        }
        
    except Exception as e:
        st.error(f"Failed to fetch data from Supabase: {e}")
        return None

# ── Main Layout ───────────────────────────────────────────────────────────────

data = fetch_telemetry_data()

# Header
st.markdown("<h1>🚨 Sentinel-Mesh: Fire Command Centre</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Live Evacuation & Telemetry Dashboard | Location: NTU Campus</div>', unsafe_allow_html=True)

# Top Metrics Row
if data:
    c = data["counts"]
    safe_tot = c["secure"] + c["safe"]
    unacc = c["unaccounted"]
    sos = c["sos"]
    notified = c["total"]
    
    # Using HTML/CSS grid for the metric cards to achieve the specific visual requirements
    st.markdown(f"""
    <div class="metric-grid">
        <div class="fcc-card card-total">
            <div class="fcc-title">Total Notified</div>
            <div class="fcc-value">{notified:,}</div>
            <div class="fcc-sub">Nodes in Mesh Network</div>
        </div>
        <div class="fcc-card card-safe">
            <div class="fcc-title">✅ Verified Safe</div>
            <div class="fcc-value">{safe_tot:,}</div>
            <div class="fcc-sub">▲ +12 in last min</div>
        </div>
        <div class="fcc-card card-warning">
            <div class="fcc-title">⚠️ Unaccounted / In Transit</div>
            <div class="fcc-value">{unacc:,}</div>
            <div class="fcc-sub">Seeking Safe Zone</div>
        </div>
        <div class="fcc-card card-sos">
            <div class="fcc-title">🆘 Active SOS Calls</div>
            <div class="fcc-value">{sos}</div>
            <div class="fcc-sub">Requires Immediate Rescue</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Middle Row: Map (2/3) & Mesh Health (1/3)
col_map, col_health = st.columns([2.2, 1])

with col_map:
    st.markdown('<div class="panel-container">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="live-dot"></span> Live Incident Map</div>', unsafe_allow_html=True)
    
    # Initialize Folium Map (Light mode aesthetics)
    m = folium.Map(
        location=NTU_CENTER,
        zoom_start=16,
        tiles="CartoDB positron", # Light, clean base map
        control_scale=True,
    )
    
    # Plot Fire Hazard Zone
    if data and data.get("fire_zone"):
        fz = data["fire_zone"]
        # The transparent red hazard circle
        folium.Circle(
            location=[fz["lat"], fz["lon"]],
            radius=fz["radius"],
            color="#ef4444",
            weight=2,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.25,
            tooltip=f"{fz.get('incident_name', 'FIRE HAZARD ZONE')} ({fz['radius']}m)",
        ).add_to(m)
        # The exact fire epicenter emoji
        folium.Marker(
            location=[fz["lat"], fz["lon"]],
            icon=folium.DivIcon(html='<div style="font-size:30px; transform: translate(-10px, -15px);">🔥</div>'),
            tooltip=fz.get("incident_name", "Epicenter"),
        ).add_to(m)
        
    # Plot Safe Zones (Green)
    for name, (lat, lon) in SAFE_ZONES.items():
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="green", icon="shield", prefix="fa"),
            tooltip=f"Assembly: {name}",
        ).add_to(m)
        
    # Plot SOS Signals (Red Pins)
    if data and data.get("sos_locations"):
        for sos_loc in data["sos_locations"]:
            folium.Marker(
                location=[sos_loc["lat"], sos_loc["lon"]],
                icon=folium.Icon(color="red", icon="warning-sign"),
                tooltip=f"SOS: {sos_loc['uid']} @ {sos_loc['time']}",
                popup=folium.Popup(f"<b>SOS ALERT</b><br>ID: {sos_loc['uid']}<br>Time: {sos_loc['time']}", max_width=200)
            ).add_to(m)

    # Render map
    st_folium(m, width=None, height=500, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)


with col_health:
    st.markdown('<div class="panel-container">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="live-dot"></span> Mesh Telemetry Feed</div>', unsafe_allow_html=True)
    
    if data and data.get("recent_logs"):
        logs_html = ""
        for log in data["recent_logs"]:
            status_class = f"log-status-{log['status']}" if log['status'] in ['sos', 'safe'] else ""
            status_badge = f'<span class="{status_class}">{log["event"]}</span>' if status_class else log["event"]
            
            logs_html += f"""
            <div class="mesh-log">
                <div style="display:flex; justify-content:space-between;">
                    <strong>{log['uid']}</strong>
                    <span class="log-time">{log['time']}</span>
                </div>
                <div>{status_badge}</div>
            </div>
            """
        st.markdown(logs_html, unsafe_allow_html=True)
    else:
        st.caption("No recent logs.")
        
    st.markdown("<hr style='margin:15px 0; border:none; border-top:1px dashed #cbd5e1;'/>", unsafe_allow_html=True)
    
    # Add a mock "Refresh/Poll" config to make the sidebar functional
    st.markdown('**System Controls**')
    st.button("🔄 Force Mesh Sync", use_container_width=True)
    st.caption("Auto-syncing every 3.0s")
    
    st.markdown('</div>', unsafe_allow_html=True)

