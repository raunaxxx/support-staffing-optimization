"""SimPy queue for a multi-server support channel with abandonment.

Capacity changes by the hour so a shift roster can be validated against
the same arrivals. The analytical model (Erlang C) assumes nobody hangs
up; this simulation is where patience shows up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from erlang import (
    CHANNEL_PARAMS,
    CHANNELS,
    DATA,
    SHRINKAGE,
    logged_in_from_rostered,
    lognormal_mu,
)

ROOT = Path(__file__).resolve().parents[1]


class AgentPool:
    """Server pool whose capacity can change without dropping in-progress work."""

    def __init__(self, env: simpy.Environment, capacity: int) -> None:
        self.env = env
        self.capacity = capacity
        self.busy = 0
        self.waiting: list[simpy.Event] = []

    def set_capacity(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self._dispatch()

    def _dispatch(self) -> None:
        while self.waiting and self.busy < self.capacity:
            event = self.waiting.pop(0)
            if not event.triggered:
                self.busy += 1
                event.succeed()

    def enqueue(self, event: simpy.Event) -> None:
        if self.busy < self.capacity and not self.waiting:
            self.busy += 1
            event.succeed()
        else:
            self.waiting.append(event)

    def cancel(self, event: simpy.Event) -> None:
        if event in self.waiting:
            self.waiting.remove(event)

    def release(self) -> None:
        self.busy = max(0, self.busy - 1)
        self._dispatch()


def simulate_queue(
    arrivals_sec: np.ndarray,
    handle_sec: np.ndarray,
    patience_sec: np.ndarray,
    capacity_changes: list[tuple[float, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one channel.

    Parameters are parallel arrays. ``capacity_changes`` is a sorted list of
    ``(time_sec, rostered_logged_in_agents)``. Times are seconds from a shared origin.

    Returns wait seconds and an abandoned flag, aligned to the input order.
    """
    arrivals_sec = np.asarray(arrivals_sec, dtype=float)
    handle_sec = np.asarray(handle_sec, dtype=float)
    patience_sec = np.asarray(patience_sec, dtype=float)
    n = len(arrivals_sec)
    waits = np.zeros(n, dtype=float)
    abandoned = np.zeros(n, dtype=np.int8)
    if n == 0:
        return waits, abandoned

    order = np.argsort(arrivals_sec, kind="mergesort")
    arrivals_sec = arrivals_sec[order]
    handle_sec = handle_sec[order]
    patience_sec = patience_sec[order]

    env = simpy.Environment()
    initial = 0
    later: list[tuple[float, int]] = []
    for time_sec, agents in capacity_changes:
        if time_sec <= 0.0 and not later:
            initial = int(agents)
        else:
            later.append((float(time_sec), int(agents)))
    pool = AgentPool(env, initial)

    def apply_capacity() -> simpy.events.Process:
        for time_sec, agents in later:
            yield env.timeout(time_sec - env.now)
            pool.set_capacity(agents)

    def customer(position: int) -> simpy.events.Process:
        start = env.now
        granted = env.event()
        pool.enqueue(granted)
        if not granted.triggered:
            patience = float(patience_sec[position])
            yield granted | env.timeout(patience)
            if not granted.triggered:
                pool.cancel(granted)
                waits_out[position] = env.now - start
                abandoned_out[position] = 1
                return
        wait = env.now - start
        yield env.timeout(float(handle_sec[position]))
        pool.release()
        waits_out[position] = wait
        abandoned_out[position] = 0

    def source() -> simpy.events.Process:
        previous = 0.0
        for position, arrival in enumerate(arrivals_sec):
            delay = float(arrival) - previous
            if delay > 0.0:
                yield env.timeout(delay)
            previous = float(arrival)
            env.process(customer(position))

    # Filled in customer(); indexed in sorted order, then unscrambled.
    waits_out = np.zeros(n, dtype=float)
    abandoned_out = np.zeros(n, dtype=np.int8)

    if later:
        env.process(apply_capacity())
    env.process(source())
    env.run()

    waits[order] = waits_out
    abandoned[order] = abandoned_out
    return waits, abandoned


def hourly_capacity_changes(
    rostered_by_hour: np.ndarray,
    shrinkage: float = SHRINKAGE,
) -> list[tuple[float, int]]:
    """Turn a per-hour roster into logged-in capacity change-points.

    ``rostered_by_hour[t]`` is paid headcount during hour ``t`` from the origin.
    """
    logged_in = np.array(
        [logged_in_from_rostered(int(n), shrinkage) for n in rostered_by_hour],
        dtype=int,
    )
    changes: list[tuple[float, int]] = []
    for hour, agents in enumerate(logged_in):
        if hour == 0 or agents != logged_in[hour - 1]:
            changes.append((hour * 3600.0, int(agents)))
    return changes


def coverage_by_dow(
    shifts: pd.DataFrame, channel: str, shift_hours: int = 8
) -> np.ndarray:
    """Weekly template coverage. Shape (7, 24), Monday = 0."""
    coverage = np.zeros((7, 24), dtype=int)
    subset = shifts[shifts["channel"] == channel]
    for dow, start, agents in zip(subset["dow"], subset["start_hour"], subset["agents"]):
        for step in range(shift_hours):
            hour = (int(start) + step) % 24
            day = (int(dow) + (int(start) + step) // 24) % 7
            coverage[day, hour] += int(agents)
    return coverage


def _metrics(
    waits: np.ndarray,
    abandoned: np.ndarray,
    handle_sec: np.ndarray,
    sla_sec: float,
    logged_in_hours: float,
) -> dict[str, float]:
    offered = len(waits)
    handled_mask = abandoned == 0
    handled = int(handled_mask.sum())
    within = int(((abandoned == 0) & (waits <= sla_sec)).sum())
    asa = float(waits[handled_mask].mean()) if handled else 0.0
    handle_hours = float(handle_sec[handled_mask].sum()) / 3600.0
    occupancy = handle_hours / logged_in_hours if logged_in_hours else 0.0
    return {
        "offered": offered,
        "handled": handled,
        "abandoned": int(offered - handled),
        "within_sla": within,
        "sla_pct": 100.0 * within / offered if offered else 0.0,
        "asa_sec": asa,
        "abandon_pct": 100.0 * (offered - handled) / offered if offered else 0.0,
        "occupancy": occupancy,
        "logged_in_hours": logged_in_hours,
    }


def simulate_week(
    forecast: pd.DataFrame,
    coverage: np.ndarray,
    channel: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One Poisson week of ``channel`` against a (7, 24) rostered-agent template."""
    params = CHANNEL_PARAMS[channel]
    sla_sec = float(params["sla_sec"])
    sigma = float(params["aht_sigma"])
    patience_mean = float(params["patience_sec"])
    channel_forecast = forecast[forecast["channel"] == channel]

    arrivals: list[float] = []
    handles: list[float] = []
    patiences: list[float] = []
    for dow in range(7):
        for hour in range(24):
            match = channel_forecast[
                (channel_forecast["dow"] == dow) & (channel_forecast["hour"] == hour)
            ]
            if match.empty:
                continue
            lam = float(match["avg_arrivals"].iloc[0])
            aht = float(match["aht_sec"].iloc[0])
            count = int(rng.poisson(lam)) if lam > 0 else 0
            if count == 0:
                continue
            origin = (dow * 24 + hour) * 3600.0
            arrivals.extend(origin + rng.uniform(0.0, 3600.0, size=count))
            mu = lognormal_mu(max(aht, 1.0), sigma)
            draws = rng.lognormal(mu, sigma, size=count)
            handles.extend(np.clip(draws, 15.0, 7200.0))
            patiences.extend(np.maximum(1.0, rng.exponential(patience_mean, size=count)))

    flat = coverage.reshape(-1)
    waits, abandoned_flag = simulate_queue(
        np.asarray(arrivals),
        np.asarray(handles),
        np.asarray(patiences),
        hourly_capacity_changes(flat),
    )
    logged_in_hours = float(
        sum(logged_in_from_rostered(int(n)) for n in flat)
    )
    metrics = _metrics(
        waits, abandoned_flag, np.asarray(handles, dtype=float), sla_sec, logged_in_hours
    )
    metrics["scenario_agents_peak"] = int(coverage.max()) if coverage.size else 0
    return metrics


def _combine_replications(reps: list[dict[str, float]]) -> dict[str, float]:
    offered = sum(m["offered"] for m in reps)
    handled = sum(m["handled"] for m in reps)
    abandoned = sum(m["abandoned"] for m in reps)
    within = sum(m["within_sla"] for m in reps)
    logged_in_hours = sum(m["logged_in_hours"] for m in reps)
    handle_hours = sum(m["occupancy"] * m["logged_in_hours"] for m in reps)
    asa = (
        sum(m["asa_sec"] * m["handled"] for m in reps) / handled if handled else 0.0
    )
    return {
        "offered": offered,
        "handled": handled,
        "abandoned": abandoned,
        "within_sla": within,
        "sla_pct": 100.0 * within / offered if offered else 0.0,
        "asa_sec": asa,
        "abandon_pct": 100.0 * abandoned / offered if offered else 0.0,
        "occupancy": handle_hours / logged_in_hours if logged_in_hours else 0.0,
        "logged_in_hours": logged_in_hours,
        "replications": float(len(reps)),
    }


def main() -> None:
    forecast_path = DATA / "demand_forecast.csv"
    baseline_path = DATA / "baseline_shifts.csv"
    optimal_path = DATA / "optimal_shifts.csv"
    for path in (forecast_path, baseline_path, optimal_path):
        if not path.exists():
            raise SystemExit(
                f"Missing {path.relative_to(ROOT)}. Run the earlier pipeline steps first."
            )

    forecast = pd.read_csv(forecast_path)
    baseline_shifts = pd.read_csv(baseline_path)
    optimal_shifts = pd.read_csv(optimal_path)
    replications = 3

    rows: list[dict[str, float | str]] = []
    for channel in CHANNELS:
        base_cov = coverage_by_dow(baseline_shifts, channel)
        opt_cov = coverage_by_dow(optimal_shifts, channel)
        for scenario, grid in (("baseline", base_cov), ("optimal", opt_cov)):
            reps = [
                simulate_week(forecast, grid, channel, np.random.default_rng(1000 + seed))
                for seed in range(replications)
            ]
            metrics = _combine_replications(reps)
            metrics["channel"] = channel
            metrics["scenario"] = scenario
            metrics["sla_target_pct"] = float(CHANNEL_PARAMS[channel]["sla_pct"]) * 100.0
            rows.append(metrics)
            print(
                f"{channel:5} {scenario:8}  SLA {metrics['sla_pct']:5.1f}%"
                f"  (target {metrics['sla_target_pct']:.0f}%)"
                f"  ASA {metrics['asa_sec']:6.1f}s"
                f"  abandon {metrics['abandon_pct']:4.1f}%"
                f"  occ {metrics['occupancy'] * 100:4.1f}%"
            )

    summary = pd.DataFrame(rows)
    front = ["channel", "scenario", "sla_pct", "sla_target_pct", "asa_sec", "abandon_pct", "occupancy", "offered", "handled", "abandoned"]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    out = DATA / "sim_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
