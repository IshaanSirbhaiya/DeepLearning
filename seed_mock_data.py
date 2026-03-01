import os
from dotenv import load_dotenv
from supabase import create_client, Client
import random
import time

load_dotenv(".streamlit/secrets.toml")

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("Please set SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml")
    exit(1)

supabase: Client = create_client(url, key)

print("Connected to Supabase. Attempting to seed mock data into table 'users'...")

# Note: In Supabase, you usually create the table in the SQL Editor. 
# We'll assume the table `users` is created with columns: id, status, lat, lon, created_at, updated_at

statuses = ["secure", "safe", "unaccounted", "sos"]
base_lat, base_lon = 1.3453, 103.6815

dummy_users = []
for i in range(50):
    status = random.choices(statuses, weights=[60, 20, 15, 5])[0]
    
    # Scatter locations around NTU center
    lat = base_lat + random.uniform(-0.005, 0.005)
    lon = base_lon + random.uniform(-0.005, 0.005)
    
    dummy_users.append({
        "status": status,
        "lat": lat if status == 'sos' else None, # only sos needs explicit mapping in this mock
        "lon": lon if status == 'sos' else None
    })

try:
    data, count = supabase.table('users').insert(dummy_users).execute()
    print(f"Successfully inserted {len(dummy_users)} mock users.")
except Exception as e:
    print(f"Failed to insert data: {e}")
    print("\nDid you create the `users` table in Supabase?")
    print("Run this SQL in your Supabase project:")
    print("CREATE TABLE users (id UUID DEFAULT uuid_generate_v4() PRIMARY KEY, status TEXT, lat FLOAT, lon FLOAT, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW());")

