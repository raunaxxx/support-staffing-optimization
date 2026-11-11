"""Erlang C staffing and workload conversion for support channels.

Phone and chat use Erlang C: the smallest logged-in pool that hits the
service-level target. Email is asynchronous, so it uses an occupancy
workload rule instead of a seconds-level delay target.

Rostered headcount is logged-in headcount grossed up for shrinkage
(breaks, meetings, absenteeism), because Erlang C describes people
actually available to take contacts.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Shrinkage is paid time when an agent is not available to take contacts.
SHRINKAGE = 0.30
EMAIL_OCCUPANCY = 0.85
# Below this offered load, email waits for the next staffed hour.
EMAIL_MIN_ERLANGS = 0.50

# Planning parameters. Volumes in the historical file are estimated;
# these drive service targets, handle-time shape, and patience.
CHANNEL_PARAMS: dict[str, dict[str, float | str]] = {
    "phone": {
        "aht_sec": 360.0,
        "aht_sigma": 0.45,
        "patience_sec": 180.0,
        "sla_sec": 20.0,
        "sla_pct": 0.80,
        "method": "erlang_c",
        "hourly_cost": 28.0,
    },
    "chat": {
        "aht_sec": 480.0,
        "aht_sigma": 0.50,
        "patience_sec": 240.0,
        "sla_sec": 60.0,
        "sla_pct": 0.80,
        "method": "erlang_c",
        "hourly_cost": 24.0,
    },
    "email": {
        "aht_sec": 540.0,
        "aht_sigma": 0.55,
        "patience_sec": 8 * 3600.0,
        "sla_sec": 4 * 3600.0,
        "sla_pct": 0.95,
        "method": "workload",
        "hourly_cost": 22.0,
    },
}

CHANNELS = tuple(CHANNEL_PARAMS)


def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def erlang_c(servers: int, traffic: float) -> float:
    """Probability an arrival waits (Erlang C). Traffic is in Erlangs.

    Returns 1 when the system is unstable (traffic >= servers).
    """
    if traffic <= 0.0 or servers <= 0:
        return 0.0
    if traffic >= servers:
        return 1.0

    log_term = [0.0]
    for k in range(1, servers + 1):
        log_term.append(log_term[-1] + math.log(traffic) - math.log(k))

    # P(wait) = [a^n / n! * n / (n - a)] / [sum_{k<n} a^k/k! + that term]
    log_b = _logsumexp(log_term[:-1])
    log_num = log_term[servers] + math.log(servers) - math.log(servers - traffic)
    return 1.0 / (math.exp(log_b - log_num) + 1.0)


def service_level(servers: int, traffic: float, target_sec: float, aht_sec: float) -> float:
    """P(wait <= target_sec) for an M/M/n queue with no abandonment."""
    if traffic <= 0.0:
        return 1.0
    if servers <= traffic or aht_sec <= 0.0:
        return 0.0
    wait_prob = erlang_c(servers, traffic)
    decay = math.exp(-(servers - traffic) * target_sec / aht_sec)
    return 1.0 - wait_prob * decay


def average_speed_of_answer(servers: int, traffic: float, aht_sec: float) -> float:
    """Mean wait in seconds, including contacts that do not wait."""
    if traffic <= 0.0 or servers <= traffic or aht_sec <= 0.0:
        return 0.0
    return erlang_c(servers, traffic) * aht_sec / (servers - traffic)


def agents_required(
    traffic: float,
    target_sec: float,
    aht_sec: float,
    sl_target: float,
    max_servers: int = 400,
) -> int:
    """Smallest logged-in pool whose Erlang C service level meets the target."""
    if traffic <= 0.0:
        return 0
    servers = math.floor(traffic) + 1
    while servers < max_servers:
        if service_level(servers, traffic, target_sec, aht_sec) >= sl_target:
            return servers
        servers += 1
    return servers


def rostered_from_logged_in(logged_in: int, shrinkage: float = SHRINKAGE) -> int:
    if logged_in <= 0:
        return 0
    return math.ceil(logged_in / (1.0 - shrinkage))


def logged_in_from_rostered(rostered: int, shrinkage: float = SHRINKAGE) -> int:
    """Agents expected on the phone after shrinkage. Does not preempt a shift."""
    if rostered <= 0:
        return 0
    return math.floor(rostered * (1.0 - shrinkage) + 1e-9)


def lognormal_mu(mean: float, sigma: float) -> float:
    """μ such that LogNormal(μ, σ) has the requested arithmetic mean."""
    return math.log(mean) - 0.5 * sigma * sigma


def _staff_row(row: pd.Series) -> dict[str, float | int | str]:
    traffic = float(row["erlangs"])
    aht = float(row["aht_sec"])
    method = str(row["method"])
    if method == "erlang_c":
        logged_in = agents_required(
            traffic, float(row["sla_sec"]), aht, float(row["sla_pct"])
        )
        predicted_sl = (
            service_level(logged_in, traffic, float(row["sla_sec"]), aht)
            if logged_in
            else 1.0
        )
        predicted_asa = average_speed_of_answer(logged_in, traffic, aht) if logged_in else 0.0
    else:
        if traffic < EMAIL_MIN_ERLANGS:
            logged_in = 0
        else:
            logged_in = math.ceil(traffic / EMAIL_OCCUPANCY)
        predicted_sl = float("nan")
        predicted_asa = float("nan")

    rostered = rostered_from_logged_in(logged_in)
    occupancy = traffic / logged_in if logged_in else 0.0
    return {
        "logged_in_required": logged_in,
        "rostered_required": rostered,
        "predicted_sl": predicted_sl,
        "predicted_asa_sec": predicted_asa,
        "occupancy": occupancy,
    }


def build_requirements(forecast: pd.DataFrame) -> pd.DataFrame:
    staffed = forecast.apply(_staff_row, axis=1, result_type="expand")
    out = pd.concat([forecast.reset_index(drop=True), staffed], axis=1)
    out["dow_name"] = out["dow"].map(lambda d: DOW_NAMES[int(d)])
    out["interval_key"] = (
        out["channel"].astype(str)
        + "-"
        + out["dow"].astype(int).astype(str)
        + "-"
        + out["hour"].astype(int).astype(str)
    )
    return out


def _self_check() -> None:
    # M/M/1 with utilisation 0.5: probability of waiting equals utilisation.
    got = erlang_c(1, 0.5)
    if abs(got - 0.5) > 1e-9:
        raise RuntimeError(f"Erlang C self-check failed: {got}")
    sl = service_level(1, 0.5, 0.0, 1.0)
    if abs(sl - 0.5) > 1e-9:
        raise RuntimeError(f"Service level self-check failed: {sl}")


def main() -> None:
    _self_check()
    forecast_path = DATA / "demand_forecast.csv"
    if not forecast_path.exists():
        raise SystemExit("Missing data/demand_forecast.csv. Run src/fit_demand.py first.")

    forecast = pd.read_csv(forecast_path)
    requirements = build_requirements(forecast)
    out = DATA / "staffing_requirements.csv"
    requirements.to_csv(out, index=False)

    peak = (
        requirements.sort_values("rostered_required", ascending=False)
        .groupby("channel", as_index=False)
        .head(1)[
            [
                "channel",
                "dow_name",
                "hour",
                "avg_arrivals",
                "erlangs",
                "logged_in_required",
                "rostered_required",
                "predicted_sl",
            ]
        ]
    )
    print(f"Wrote {out}")
    print("Peak rostered hour by channel:")
    print(peak.to_string(index=False))


if __name__ == "__main__":
    main()
