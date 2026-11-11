-- Schema for the support staffing database.
-- src/gen_data.py runs this script, then loads agents, shifts, and contacts.
-- Running it on its own drops those tables.

DROP VIEW IF EXISTS v_sla_summary;
DROP VIEW IF EXISTS v_sla;
DROP VIEW IF EXISTS v_hourly_arrivals;

DROP TABLE IF EXISTS shifts;
DROP TABLE IF EXISTS contacts;
DROP TABLE IF EXISTS agents;

CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    channel TEXT NOT NULL,
    hourly_cost REAL NOT NULL,
    contract TEXT NOT NULL
);

CREATE TABLE shifts (
    shift_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents (agent_id),
    channel TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    hourly_cost REAL NOT NULL
);

CREATE TABLE contacts (
    contact_id INTEGER PRIMARY KEY,
    arrived_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    priority TEXT NOT NULL,
    handle_time_sec REAL NOT NULL,
    patience_sec REAL NOT NULL,
    wait_time_sec REAL NOT NULL,
    abandoned INTEGER NOT NULL,
    sla_sec INTEGER NOT NULL,
    within_sla INTEGER NOT NULL
);

CREATE INDEX idx_contacts_arrived ON contacts (channel, arrived_at);
CREATE INDEX idx_shifts_start ON shifts (channel, start_time);
