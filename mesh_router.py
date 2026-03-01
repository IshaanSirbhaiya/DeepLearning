import requests
import osmnx as ox
import networkx as nx
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import math

# --- 1. CONFIGURATION & CREDENTIALS ---
TELEGRAM_BOT_TOKEN = "8622784083:AAG9eU9XMSZQJ0_MC90RYcmqn-P31FAKiPE"
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# UPDATED: Both IDs now in the active broadcast list
REGISTERED_USERS = ["5808527465", "1674734061","5574113481"]
SUPABASE_URL = "https://jyuxqhkutlblnzrutuey.supabase.co"
SUPABASE_KEY = "sb_publishable_KuMdlRMDYfFELn8br7zYWQ_5PRFJsU6"

# --- 2. ENGINE MATH ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# --- 3. SYSTEM SETUP (DYNAMIC PERIMETERS) ---
print("1. Loading NTU Map Engine...")
ntu_center = (1.3460, 103.6810)
G = ox.graph_from_point(ntu_center, dist=1500, network_type='walk')

# The exact fire coordinates you copied from the database
fire_lng, fire_lat = 103.68275, 1.3432

print("2. Generating Dynamic Safe Perimeters...")
# 500 meters is approximately 0.0045 degrees
offset = 0.0045 

# Project 4 points 500m away in all cardinal directions
dynamic_safe_points = {
    "North Evac Point": (fire_lng, fire_lat + offset),
    "South Evac Point": (fire_lng, fire_lat - offset),
    "East Evac Point": (fire_lng + offset, fire_lat),
    "West Evac Point": (fire_lng - offset, fire_lat)
}

safe_zones = {}
for name, coords in dynamic_safe_points.items():
    nearest_node = ox.distance.nearest_nodes(G, X=coords[0], Y=coords[1])
    n_lat = G.nodes[nearest_node]['y']
    n_lng = G.nodes[nearest_node]['x']
    
    # Ensure the newly generated point is actually outside the 350m danger zone
    if calculate_distance(fire_lat, fire_lng, n_lat, n_lng) > 350:
        safe_zones[name] = {
            "coords": (n_lng, n_lat),
            "node": nearest_node
        }
print(f"-> Secured {len(safe_zones)} dynamic rally points around the hazard.")

print("3. Establishing 80-meter Quarantine Blast Radius...")
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
    
    # IMPROVED NAME DETECTION
    first = message.from_user.first_name if message.from_user.first_name else ""
    last = message.from_user.last_name if message.from_user.last_name else ""
    u_name = f"{first} {last}".strip()
    
    if not u_name:
        u_name = f"User_{message.from_user.id}"
    
    chat_id = message.chat.id
    dist = calculate_distance(fire_lat, fire_lng, u_lat, u_lng)
    
    status = "endangered" if dist <= 350 else "secure"
    
    # GENERATE CLICKABLE GOOGLE MAPS LINK FOR ADMIN DASHBOARD
    user_gmaps_link = f"https://www.google.com/maps?q={u_lat},{u_lng}"
    
    payload = {
        "chat_id": chat_id, 
        "name": u_name, 
        "status": status, 
        "location_link": user_gmaps_link
    }
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates" 
    }
    api_endpoint = f"{SUPABASE_URL}/rest/v1/evacuees"

    try:
        print(f"DEBUG: Pushing {u_name} via Direct REST API...")
        response = requests.post(api_endpoint, json=payload, headers=headers)
        
        if response.status_code in [200, 201]:
            print(f"✅ Success: {u_name} synced to Command Center!")
        else:
            print(f"❌ API Rejected: {response.text}")
            
    except Exception as e:
        print(f"❌ Critical Network Error: {e}")

    # Evacuation Response
    hide_markup = telebot.types.ReplyKeyboardRemove()
    if status == "secure":
        bot.reply_to(message, "🟢 STATUS: SAFE. Stay clear of The Hive.", reply_markup=hide_markup)
    else:
        u_node = ox.distance.nearest_nodes(G, X=u_lng, Y=u_lat)
        best_route, best_zone = None, ""
        shortest_dist = float('inf')
        
        # This will now automatically search the 4 new dynamic points!
        for name, data in safe_zones.items():
            try:
                d = nx.shortest_path_length(G, source=u_node, target=data["node"], weight='length')
                if d < shortest_dist:
                    shortest_dist, best_route, best_zone = d, nx.shortest_path(G, source=u_node, target=data["node"], weight='length'), name
            except: continue

        if best_route:
            step = max(1, len(best_route) // 4)
            waypoints = "|".join([f"{G.nodes[best_route[i]]['y']},{G.nodes[best_route[i]]['x']}" for i in range(step, len(best_route)-1, step)])
            s_lng, s_lat = safe_zones[best_zone]["coords"]
            gmaps_link = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lng}&destination={s_lat},{s_lng}&waypoints={waypoints}&travelmode=walking"
            bot.reply_to(message, f"🔴 ENDANGERED. Proceed to {best_zone}.\n📍 <a href='{gmaps_link}'>Safe Route</a>", parse_mode="HTML", reply_markup=hide_markup)

# --- 5. EXECUTION: Mass Alert Everyone ---
def mass_alert():
    print("4. Broadcasting alert to ALL registered users...")
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add(KeyboardButton("📍 SEND MY LOCATION", request_location=True))
    
    for user_id in REGISTERED_USERS:
        try:
            bot.send_message(user_id, "🚨 <b>CRITICAL ALARM</b> 🚨\nFire at The Hive! Tap below immediately.", parse_mode="HTML", reply_markup=markup)
            print(f"✅ Alert sent to: {user_id}")
        except Exception as e:
            print(f"❌ Could not reach {user_id}: {e}")

if __name__ == "__main__":
    mass_alert()
    print("🤖 Bot Online & Listening for all teammate locations...")
    bot.infinity_polling()