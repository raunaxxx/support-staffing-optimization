-- Mean offered load by channel, weekday, and hour.
-- dow is Monday = 0, matching the Python pipeline.
-- Hours with no contacts are omitted; demand_forecast.csv keeps the full grid.

DROP VIEW IF EXISTS v_hourly_arrivals;

CREATE VIEW v_hourly_arrivals AS
WITH days AS (
    SELECT
        (CAST(strftime('%w', arrived_at) AS INTEGER) + 6) % 7 AS dow,
        COUNT(DISTINCT date(arrived_at)) AS sample_days
    FROM contacts
    GROUP BY dow
),
hourly AS (
    SELECT
        channel,
        (CAST(strftime('%w', arrived_at) AS INTEGER) + 6) % 7 AS dow,
        CAST(strftime('%H', arrived_at) AS INTEGER) AS hour,
        COUNT(*) AS total_contacts,
        AVG(CASE WHEN abandoned = 0 THEN handle_time_sec END) AS aht_sec
    FROM contacts
    GROUP BY channel, dow, hour
)
SELECT
    h.channel,
    h.dow,
    h.hour,
    d.sample_days,
    ROUND(h.total_contacts * 1.0 / d.sample_days, 2) AS avg_arrivals,
    ROUND(h.aht_sec, 1) AS aht_sec,
    ROUND(h.total_contacts * 1.0 / d.sample_days * h.aht_sec / 3600.0, 2) AS erlangs,
    h.total_contacts
FROM hourly AS h
JOIN days AS d ON d.dow = h.dow
ORDER BY h.channel, h.dow, h.hour;
