-- Run this SQL in your Supabase SQL Editor to set up the full schema.

-- ── Evacuees table ────────────────────────────────────────────────────────────
CREATE TABLE evacuees (
    id SERIAL PRIMARY KEY,
    name TEXT,
    chat_id TEXT,
    status TEXT,             -- 'secure' | 'safe' | 'unaccounted' | 'sos' | 'endangered'
    location_link TEXT,      -- Google Maps URL e.g. https://maps.google.com/maps?q=1.344,103.682
    last_update TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE evacuees ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public can read evacuees"   ON evacuees FOR SELECT USING (true);
CREATE POLICY "Public can insert evacuees" ON evacuees FOR INSERT WITH CHECK (true);
CREATE POLICY "Public can update evacuees" ON evacuees FOR UPDATE USING (true);

-- ── Hazards table ─────────────────────────────────────────────────────────────
CREATE TABLE hazards (
    id SERIAL PRIMARY KEY,
    name TEXT,                            -- e.g., "The Hive Fire"
    latitude FLOAT,
    longitude FLOAT,
    status TEXT DEFAULT 'active',         -- 'active' | 'contained'
    reported_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE hazards ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow everything for demo" ON public.hazards
    FOR ALL TO public USING (true) WITH CHECK (true);

-- Example insert:
-- INSERT INTO hazards (name, latitude, longitude, status)
-- VALUES ('The Hive Fire', 1.3437, 103.6801, 'active');
