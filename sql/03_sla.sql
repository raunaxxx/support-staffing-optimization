-- Service level = contacts answered within the channel target / offered.
-- Abandoned contacts stay in the denominator, so they count as a miss.
-- Phone target 20 seconds, chat 60 seconds, email 4 hours (stored on each row).

DROP VIEW IF EXISTS v_sla_summary;
DROP VIEW IF EXISTS v_sla;

CREATE VIEW v_sla AS
SELECT
    channel,
    date(arrived_at) AS service_date,
    (CAST(strftime('%w', arrived_at) AS INTEGER) + 6) % 7 AS dow,
    CAST(strftime('%H', arrived_at) AS INTEGER) AS hour,
    COUNT(*) AS offered,
    SUM(CASE WHEN abandoned = 0 THEN 1 ELSE 0 END) AS handled,
    SUM(abandoned) AS abandoned,
    SUM(within_sla) AS within_sla,
    ROUND(100.0 * SUM(within_sla) / COUNT(*), 1) AS sla_pct,
    ROUND(AVG(CASE WHEN abandoned = 0 THEN wait_time_sec END), 1) AS asa_sec,
    ROUND(100.0 * SUM(abandoned) / COUNT(*), 1) AS abandon_pct
FROM contacts
GROUP BY channel, service_date, dow, hour;

CREATE VIEW v_sla_summary AS
SELECT
    channel,
    COUNT(*) AS offered,
    SUM(CASE WHEN abandoned = 0 THEN 1 ELSE 0 END) AS handled,
    SUM(abandoned) AS abandoned,
    ROUND(100.0 * SUM(within_sla) / COUNT(*), 1) AS sla_pct,
    ROUND(AVG(CASE WHEN abandoned = 0 THEN wait_time_sec END), 1) AS asa_sec,
    ROUND(100.0 * SUM(abandoned) / COUNT(*), 1) AS abandon_pct,
    ROUND(AVG(CASE WHEN abandoned = 0 THEN handle_time_sec END), 1) AS aht_sec
FROM contacts
GROUP BY channel;
