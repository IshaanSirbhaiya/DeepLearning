"""
mesh_router.py — SafeEdge Evacuation Router + Telegram Bot

Reads fire location from the detection layer's /fire endpoint (port 8001)
or falls back to Supabase hazards table. Routes users to nearest safe zone
via NetworkX shortest path on NTU campus walking network.
"""

import os
import requests
import osmnx as ox
import networkx as nx
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import math
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

# --- 1. CONFIGURATION & CREDENTIALS ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None

REGISTERED_USERS = ["5808527465", "1674734061", "5574113481", "8112299954"]
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

DETECTION_API = os.getenv("DETECTION_API", "http://localhost:8001")

# 9 Official NTU Assembly Areas
SAFE_ZONES = {
    "North Spine Plaza":          (1.3468, 103.6810),
    "South Spine Plaza":          (1.3425, 103.6832),
    "Sports & Rec Centre (SRC)":  (1.3496, 103.6835),
    "Yunnan Garden (Open Field)": (1.3458, 103.6858),
    "Hall 1-3 Field":             (1.3540, 103.6855),
    "Innovation Centre Carpark":  (1.3448, 103.6785),
    "The Arc (North Spine CP-E)": (1.3475, 103.6800),
    "CCEB Assembly (CW4)":        (1.3435, 103.6870),
    "CCDS Assembly (N3 Carpark)": (1.3460, 103.6790),
}

# --- 2. ENGINE MATH ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))


def get_fire_location():
    """
    Get fire location from detection layer or Supabase.
    Priority: 1) /fire endpoint  2) Supabase hazards  3) fallback
    """
    # Try detection layer /fire endpoint first
    try:
        resp = requests.get(f"{DETECTION_API}/fire", timeout=3)
        data = resp.json()
        if data.get("fire_detected") and "latitude" in data:
            building = data.get("location", {}).get("building", "Unknown")
            print(f"  Fire location from detection: {building} ({data['latitude']}, {data['longitude']})")
            return data["latitude"], data["longitude"], building
    except Exception:
        pass

    # Try Supabase hazards table
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/hazards?select=*&status=eq.active&order=reported_at.desc&limit=1",
            headers=headers, timeout=5,
        )
        hazards = resp.json()
        if hazards:
            h = hazards[0]
            print(f"  Fire location from Supabase: {h['name']} ({h['latitude']}, {h['longitude']})")
            return h["latitude"], h["longitude"], h["name"]
    except Exception:
        pass

    # Fallback to The Hive
    print("  Using fallback fire location: The Hive")
    return 1.34321, 103.68275, "The Hive"


# --- 3. SYSTEM SETUP ---
print("1. Loading NTU Map Engine...")
ntu_center = (1.3460, 103.6810)
G = ox.graph_from_point(ntu_center, dist=1500, network_type='walk')

print("2. Fetching fire location from detection system...")
fire_lat, fire_lng, fire_building = get_fire_location()
print(f"   -> Fire at: {fire_building} (lat={fire_lat}, lng={fire_lng})")

# Map Safe Zones to physical walking nodes on the OpenStreetMap
print("3. Mapping Safe Zones to walking network nodes...")
safe_nodes = {}
for name, coords in SAFE_ZONES.items():
    safe_nodes[name] = ox.distance.nearest_nodes(G, X=coords[1], Y=coords[0])
print(f"   -> Mapped {len(safe_nodes)} assembly areas.")

print("4. Establishing 80-meter Quarantine Blast Radius...")
danger_radius = 80
for node in G.nodes():
    n_lat, n_lng = G.nodes[node]['y'], G.nodes[node]['x']
    if calculate_distance(fire_lat, fire_lng, n_lat, n_lng) <= danger_radius:
        for neighbor in G.neighbors(node):
            if G.has_edge(node, neighbor):
                G[node][neighbor][0]['length'] = float('inf')

# --- 4. BOT LOGIC & SUPABASE SYNC ---
@bot.message_handler(content_types=['location'])
def handle_location(message):
    u_lat = message.location.latitude
    u_lng = message.location.longitude

    first = message.from_user.first_name if message.from_user.first_name else ""
    last = message.from_user.last_name if message.from_user.last_name else ""
    u_name = f"{first} {last}".strip()
    if not u_name:
        u_name = f"User_{message.from_user.id}"

    chat_id = message.chat.id
    dist = calculate_distance(fire_lat, fire_lng, u_lat, u_lng)

    status = "endangered" if dist <= 350 else "secure"

    # Database payload
    user_gmaps_link = f"https://www.google.com/maps?q={u_lat},{u_lng}"
    payload = {"chat_id": chat_id, "name": u_name, "status": status, "location_link": user_gmaps_link}
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}

    try:
        response = requests.post(f"{SUPABASE_URL}/rest/v1/evacuees", json=payload, headers=headers)
        if response.status_code in [200, 201]:
            print(f"  Success: {u_name} synced to Command Center!")
        else:
            print(f"  API Rejected: {response.text}")
    except Exception as e:
        print(f"  Network Error: {e}")

    # Interactive buttons for status updates
    interactive_buttons = InlineKeyboardMarkup()
    interactive_buttons.add(InlineKeyboardButton("I have reached Safety", callback_data="mark_safe"))
    interactive_buttons.add(InlineKeyboardButton("EMERGENCY RESCUE", callback_data="mark_emergency"))

    # Evacuation Response
    if status == "secure":
        hide_markup = telebot.types.ReplyKeyboardRemove()
        bot.reply_to(message, f"STATUS: SAFE. Stay clear of {fire_building}.", reply_markup=hide_markup)
    else:
        u_node = ox.distance.nearest_nodes(G, X=u_lng, Y=u_lat)
        best_route, best_zone = None, ""
        shortest_dist = float('inf')

        for name, node_id in safe_nodes.items():
            try:
                d = nx.shortest_path_length(G, source=u_node, target=node_id, weight='length')
                if d < shortest_dist:
                    shortest_dist = d
                    best_route = nx.shortest_path(G, source=u_node, target=node_id, weight='length')
                    best_zone = name
            except nx.NetworkXNoPath:
                continue

        if best_route:
            step = max(1, len(best_route) // 4)
            waypoints = "|".join([f"{G.nodes[best_route[i]]['y']},{G.nodes[best_route[i]]['x']}" for i in range(step, len(best_route)-1, step)])
            s_lat, s_lng = SAFE_ZONES[best_zone]
            gmaps_link = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lng}&destination={s_lat},{s_lng}&waypoints={waypoints}&travelmode=walking"

            msg_text = f"ENDANGERED.\nHazard: {fire_building}\n\nProceed immediately to <b>{best_zone}</b>.\n<a href='{gmaps_link}'>Open Safe Route</a>\n\n<i>Once you reach, press the Safe button. Click on Emergency if you need urgent help.</i>"
            bot.reply_to(message, msg_text, parse_mode="HTML", reply_markup=interactive_buttons)
        else:
            msg_text = f"ENDANGERED. Please move away from {fire_building} immediately.\n\n<i>Once you are clear, press the Safe button. Click on Emergency if you need urgent help.</i>"
            bot.reply_to(message, msg_text, parse_mode="HTML", reply_markup=interactive_buttons)


# --- BUTTON CLICK LISTENER (UPDATES DATABASE LIVE) ---
@bot.callback_query_handler(func=lambda call: call.data in ["mark_safe", "mark_emergency"])
def handle_status_buttons(call):
    chat_id = call.message.chat.id

    if call.data == "mark_safe":
        new_status = "secure"
        reply_text = "<b>STATUS UPDATED: SECURE.</b>\n\nYou have been marked as safe on the Command Dashboard. Please remain at the assembly area."
    else:
        new_status = "emergency help"
        reply_text = "<b>EMERGENCY LOGGED.</b>\n\nYour status is flashing red on the Command Dashboard. Rescue teams have been notified of your exact last known location. Stay exactly where you are."

    api_endpoint = f"{SUPABASE_URL}/rest/v1/evacuees?chat_id=eq.{chat_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.patch(api_endpoint, json={"status": new_status}, headers=headers)
        if response.status_code in [200, 204]:
            bot.answer_callback_query(call.id, "Status Updated!")
            bot.send_message(chat_id, reply_text, parse_mode="HTML")
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
            print(f"  UPDATE: User {chat_id} -> {new_status.upper()}")
        else:
            bot.answer_callback_query(call.id, "Database Sync Delayed", show_alert=True)
    except Exception:
        bot.answer_callback_query(call.id, "Network Error", show_alert=True)


# --- 5. EXECUTION: Mass Alert Everyone ---
def mass_alert():
    print("5. Broadcasting alert to ALL registered users...")
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add(KeyboardButton("SEND MY LOCATION", request_location=True))

    for user_id in REGISTERED_USERS:
        try:
            bot.send_message(user_id, f"<b>CRITICAL ALARM</b>\nFire detected at <b>{fire_building}</b>! Tap below immediately.", parse_mode="HTML", reply_markup=markup)
            print(f"  Alert sent to: {user_id}")
        except Exception as e:
            print(f"  Could not reach {user_id}: {e}")

if __name__ == "__main__":
    mass_alert()
    print("Bot Online & Listening for all teammate locations...")
    bot.infinity_polling()
