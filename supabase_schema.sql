-- Run this SQL in your Supabase SQL Editor to create the users table:

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    status TEXT NOT NULL,
    lat FLOAT,
    lon FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fire incidents table; store one row per reported fire location/update.
CREATE TABLE fire_locations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    incident_name TEXT NOT NULL DEFAULT 'Campus Fire',
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    radius_m INTEGER NOT NULL DEFAULT 100 CHECK (radius_m > 0),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    reported_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fire_locations_active_reported
    ON fire_locations (is_active, reported_at DESC);

-- Example insert:
-- INSERT INTO fire_locations (incident_name, lat, lon, radius_m, is_active)
-- VALUES ('North Spine Electrical Fire', 1.3468, 103.6810, 120, TRUE);

-- Note: In a production app you might want to enable RLS (Row Level Security),
-- but for a hackathon/dashboard prototype, it is often easier to leave this open or strictly API-key protected.
