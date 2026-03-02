import requests
import osmnx as ox
import networkx as nx
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import math
# --- 1. CONFIGURATION & CREDENTIALS ---
TELEGRAM_BOT_TOKEN = "8622784083:AAG9eU9XMSZQJ0_MC90RYcmqn-P31FAKiPE"
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
REGISTERED_USERS = ["5808527465", "1674734061","5574113481","8112299954"]
SUPABASE_URL = "https://jyuxqhkutlblnzrutuey.supabase.co"
SUPABASE_KEY = "sb_publishable_KuMdlRMDYfFELn8br7zYWQ_5PRFJsU6"
# =========================================================
# 🚨 COMMAND CENTER OVERRIDE: SET THE ACTIVE FIRE HERE 🚨
# Options: "The Hive", "North Spine", "SCBE", "Hall 2"
# =========================================================
ACTIVE_FIRE_NAME = "The Hive"
# --- 2. ENGINE MATH & HARDCODED ZONES ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))
# Precise Coordinates for the 4 Scenarios
HAZARDS = {
    "The Hive": (1.3432, 103.6827),
    "North Spine": (1.3478, 103.6800),
    "SCBE": (1.3468, 103.6836), 
    "Hall 2": (1.3463, 103.6865) 
}
# 9 Official NTU Assembly Areas
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
# --- 3. SYSTEM SETUP ---
print("1. Loading NTU Map Engine...")
ntu_center = (1.3460, 103.6810)
G = ox.graph_from_point(ntu_center, dist=1500, network_type='walk')
# Map the Safe Zones to physical walking nodes on the OpenStreetMap
safe_nodes = {}
for name, coords in SAFE_ZONES.items():
    safe_nodes[name] = ox.distance.nearest_nodes(G, X=coords[1], Y=coords[0])
# Get the coordinates for whichever fire is currently active
fire_lat, fire_lng = HAZARDS[ACTIVE_FIRE_NAME]
print(f"2. Establishing 80-meter Quarantine Blast Radius around {ACTIVE_FIRE_NAME}...")
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
    if not u_name: u_name = f"User_{message.from_user.id}"
    
    chat_id = message.chat.id
    dist = calculate_distance(fire_lat, fire_lng, u_lat, u_lng)
    
    status = "endangered" if dist <= 350 else "secure"
    
    # DATABASE PAYLOAD
    user_gmaps_link = f"https://www.google.com/maps?q={u_lat},{u_lng}"
    payload = { "chat_id": chat_id, "name": u_name, "status": status, "location_link": user_gmaps_link }
    headers = { "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates" }
    
    try:
        response = requests.post(f"{SUPABASE_URL}/rest/v1/evacuees", json=payload, headers=headers)
        if response.status_code in [200, 201]:
            print(f"✅ Success: {u_name} synced to Command Center!")
    except Exception as e:
        print(f"❌ Network Error: {e}")
    # =========================================================
    # INLINE BUTTON SETUP (SAFE & EMERGENCY)
    # =========================================================
    interactive_buttons = InlineKeyboardMarkup()
    btn_safe = InlineKeyboardButton("🟢 I have reached Safety", callback_data="mark_safe")
    btn_sos = InlineKeyboardButton("🚨 EMERGENCY RESCUE", callback_data="mark_emergency")
    interactive_buttons.add(btn_safe)
    interactive_buttons.add(btn_sos)
    # =========================================================
    # STRICT GEOGRAPHIC ISOLATION ROUTING
    # =========================================================
    if status == "secure":
        hide_markup = telebot.types.ReplyKeyboardRemove()
        bot.reply_to(message, f"🟢 STATUS: SAFE. Stay clear of {ACTIVE_FIRE_NAME}.", reply_markup=hide_markup)
    else:
        # Dynamically pick the closest safe zone to the user that is far enough from the fire
        min_safe_distance_from_fire = 200  # meters
        best_zone = None
        best_dist_to_user = float('inf')

        for zone_name, (z_lat, z_lng) in SAFE_ZONES.items():
            dist_from_fire = calculate_distance(fire_lat, fire_lng, z_lat, z_lng)
            if dist_from_fire < min_safe_distance_from_fire:
                continue  # skip zones too close to the fire
            dist_from_user = calculate_distance(u_lat, u_lng, z_lat, z_lng)
            if dist_from_user < best_dist_to_user:
                best_dist_to_user = dist_from_user
                best_zone = zone_name

        if best_zone is None:
            # Fallback: pick the zone farthest from the fire
            best_zone = max(SAFE_ZONES, key=lambda z: calculate_distance(fire_lat, fire_lng, *SAFE_ZONES[z]))

        u_node = ox.distance.nearest_nodes(G, X=u_lng, Y=u_lat)
        target_node = safe_nodes[best_zone]
        
        try:
            best_route = nx.shortest_path(G, source=u_node, target=target_node, weight='length')
            step = max(1, len(best_route) // 4)
            waypoints = "|".join([f"{G.nodes[best_route[i]]['y']},{G.nodes[best_route[i]]['x']}" for i in range(step, len(best_route)-1, step)])
            s_lat, s_lng = SAFE_ZONES[best_zone]
            
            gmaps_link = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lng}&destination={s_lat},{s_lng}&waypoints={waypoints}&travelmode=walking"
            gmaps_link_escaped = gmaps_link.replace("&", "&amp;")

            msg_text = f"🔴 ENDANGERED.\n🔥 Hazard: {ACTIVE_FIRE_NAME}\n\nProceed immediately to <b>{best_zone}</b>.\n📍 <a href='{gmaps_link_escaped}'>Open Safe Route</a>\n\n⚠️ <i>Once you reach, press the Safe button. Click on Emergency if you need urgent help.</i>"
            bot.reply_to(message, msg_text, parse_mode="HTML", reply_markup=interactive_buttons)
            
        except nx.NetworkXNoPath:
            msg_text = f"🔴 ENDANGERED. Please move away from {ACTIVE_FIRE_NAME} immediately.\n\n⚠️ <i>Once you are clear, press the Safe button. Click on Emergency if you need urgent help.</i>"
            bot.reply_to(message, msg_text, parse_mode="HTML", reply_markup=interactive_buttons)
# =========================================================
# 🚨 BUTTON CLICK LISTENER (UPDATES DATABASE LIVE) 🚨
# =========================================================
@bot.callback_query_handler(func=lambda call: call.data in ["mark_safe", "mark_emergency"])
def handle_status_buttons(call):
    chat_id = call.message.chat.id
    
    if call.data == "mark_safe":
        new_status = "secure"
        reply_text = "✅ <b>STATUS UPDATED: SECURE.</b>\n\nYou have been marked as safe on the Command Dashboard. Please remain at the assembly area."
    else:
        new_status = "emergency help"
        reply_text = "🚨 <b>EMERGENCY LOGGED.</b>\n\nYour status is flashing red on the Command Dashboard. Rescue teams have been notified of your exact last known location. Stay exactly where you are."
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
            print(f"📡 UPDATE: User {chat_id} pressed button -> {new_status.upper()}")
        else:
            bot.answer_callback_query(call.id, "Database Sync Delayed", show_alert=True)
            
    except Exception as e:
        bot.answer_callback_query(call.id, "Network Error", show_alert=True)
# --- 5. EXECUTION: Mass Alert Everyone ---
def mass_alert():
    print(f"4. Broadcasting {ACTIVE_FIRE_NAME} alert to ALL registered users...")
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add(KeyboardButton("📍 SEND MY LOCATION", request_location=True))
    
    for user_id in REGISTERED_USERS:
        try:
            bot.send_message(user_id, f"🚨 <b>CRITICAL ALARM</b> 🚨\nFire detected at <b>{ACTIVE_FIRE_NAME}</b>! Tap below immediately.", parse_mode="HTML", reply_markup=markup)
        except Exception:
            pass
if __name__ == "__main__":
    mass_alert()
    print("🤖 Bot Online & Listening for all teammate locations...")
    bot.infinity_polling()
