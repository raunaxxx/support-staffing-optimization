"""Fit a weekly demand profile from historical contacts.

Arrival rate for each channel, weekday, and hour is the mean count across
sample days of that weekday. Handle time is the mean of answered contacts.
Offered load (Erlangs) = arrivals per hour × AHT in hours.
"""

from __future__ import annotations

import math

import pandas as pd

from erlang import CHANNEL_PARAMS, CHANNELS, DATA, DOW_NAMES


def fit(contacts: pd.DataFrame) -> pd.DataFrame:
    frame = contacts.copy()
    frame["arrived_at"] = pd.to_datetime(frame["arrived_at"])
    frame["dow"] = frame["arrived_at"].dt.dayofweek
    frame["hour"] = frame["arrived_at"].dt.hour
    frame["date"] = frame["arrived_at"].dt.date

    sample_days = frame.groupby("dow")["date"].nunique().reindex(range(7), fill_value=0)
    totals = frame.groupby(["channel", "dow", "hour"]).size()
    handled = frame[frame["abandoned"] == 0]
    aht = handled.groupby(["channel", "dow", "hour"])["handle_time_sec"].mean()
    channel_aht = handled.groupby("channel")["handle_time_sec"].mean()

    rows: list[dict[str, float | int | str]] = []
    for channel in CHANNELS:
        params = CHANNEL_PARAMS[channel]
        fallback = float(channel_aht.get(channel, params["aht_sec"]))
        if math.isnan(fallback):
            fallback = float(params["aht_sec"])
        for dow in range(7):
            days = int(sample_days.loc[dow])
            for hour in range(24):
                key = (channel, dow, hour)
                total = int(totals[key]) if key in totals.index else 0
                avg = total / days if days else 0.0
                aht_sec = float(aht[key]) if key in aht.index else fallback
                if math.isnan(aht_sec):
                    aht_sec = fallback
                rows.append(
                    {
                        "channel": channel,
                        "dow": dow,
                        "dow_name": DOW_NAMES[dow],
                        "hour": hour,
                        "sample_days": days,
                        "avg_arrivals": round(avg, 4),
                        "aht_sec": round(aht_sec, 2),
                        "erlangs": round(avg * aht_sec / 3600.0, 4),
                        "sla_sec": float(params["sla_sec"]),
                        "sla_pct": float(params["sla_pct"]),
                        "method": str(params["method"]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    path = DATA / "contacts.csv"
    if not path.exists():
        raise SystemExit("Missing data/contacts.csv. Run src/gen_data.py first.")
    forecast = fit(pd.read_csv(path))
    out = DATA / "demand_forecast.csv"
    forecast.to_csv(out, index=False)
    print(f"Wrote {out} ({len(forecast)} channel-hour rows)")

    true_path = DATA / "true_hourly_rates.csv"
    if true_path.exists():
        truth = pd.read_csv(true_path)
        merged = forecast.merge(truth, on=["channel", "dow", "hour"])
        mae = (merged["avg_arrivals"] - merged["arrival_rate"]).abs().mean()
        print(f"Mean absolute error vs true hourly rate: {mae:.2f} contacts/hour")


if __name__ == "__main__":
    main()
