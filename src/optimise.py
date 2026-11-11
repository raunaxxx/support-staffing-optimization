"""Minimum-cost weekly shift roster that covers the Erlang / workload requirement.

Decision: how many 8-hour shifts to start at each hour of each weekday, per channel.
A shift may cross midnight. Shrinkage is already inside ``rostered_required``,
so one scheduled agent does not equal one logged-in agent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pulp

from erlang import CHANNEL_PARAMS, CHANNELS, DATA, DOW_NAMES, logged_in_from_rostered
from sim import coverage_by_dow

SHIFT_HOURS = 8
SHIFT_COLUMNS = [
    "channel",
    "dow",
    "dow_name",
    "start_hour",
    "agents",
    "shift_hours",
    "hourly_cost",
    "shift_cost",
]


def optimise_channel(channel: str, required: pd.DataFrame, hourly_cost: float) -> pd.DataFrame:
    """Minimum-cost 8-hour shifts whose coverage meets ``rostered_required``."""
    problem = pulp.LpProblem(f"roster_{channel}", pulp.LpMinimize)
    starts = [(dow, hour) for dow in range(7) for hour in range(24)]
    x = {
      (dow, hour): pulp.LpVariable(f"x_{channel}_{dow}_{hour}", lowBound=0, cat="Integer")
        for dow, hour in starts
    }
    problem += pulp.lpSum(x[dow, hour] * SHIFT_HOURS * hourly_cost for dow, hour in starts)

    covered: dict[tuple[int, int], list] = {(dow, hour): [] for dow in range(7) for hour in range(24)}
    for dow, start in starts:
        for step in range(SHIFT_HOURS):
            hour = (start + step) % 24
            day = (dow + (start + step) // 24) % 7
            covered[day, hour].append(x[dow, start])

    grid = (
        required.pivot(index="dow", columns="hour", values="rostered_required")
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    for dow in range(7):
        for hour in range(24):
            need = int(grid.loc[dow, hour])
            if need <= 0:
                continue
            problem += (
                pulp.lpSum(covered[dow, hour]) >= need,
                f"cov_{channel}_{dow}_{hour}",
            )

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise SystemExit(f"Roster for {channel} is {pulp.LpStatus[status]}")

    rows = []
    for dow, start in starts:
        agents = int(round(pulp.value(x[dow, start]) or 0))
        if agents <= 0:
            continue
        rows.append(
            {
                "channel": channel,
                "dow": dow,
                "dow_name": DOW_NAMES[dow],
                "start_hour": start,
                "agents": agents,
                "shift_hours": SHIFT_HOURS,
                "hourly_cost": hourly_cost,
                "shift_cost": round(agents * SHIFT_HOURS * hourly_cost, 2),
            }
        )
    if not rows:
        return pd.DataFrame(columns=SHIFT_COLUMNS)
    return pd.DataFrame(rows)


def _plot_monday(coverage: pd.DataFrame) -> None:
    monday = coverage[(coverage["channel"] == "phone") & (coverage["dow"] == 0)].sort_values("hour")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(monday["hour"], monday["logged_in_required"], label="Logged-in required", linewidth=2)
    ax.plot(monday["hour"], monday["baseline_logged_in"], label="Baseline logged-in", linestyle="--")
    ax.plot(monday["hour"], monday["optimal_logged_in"], label="Optimal logged-in", linewidth=2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Agents logged in")
    ax.set_title("Phone, Monday — required vs baseline vs optimal")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plots = DATA / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / "phone_monday_coverage.png", dpi=120)
    plt.close(fig)


def main() -> None:
    requirements_path = DATA / "staffing_requirements.csv"
    baseline_path = DATA / "baseline_shifts.csv"
    if not requirements_path.exists() or not baseline_path.exists():
        raise SystemExit("Run src/gen_data.py, src/fit_demand.py, and src/erlang.py first.")

    requirements = pd.read_csv(requirements_path)
    baseline = pd.read_csv(baseline_path)
    shift_frames = []
    for channel in CHANNELS:
        subset = requirements[requirements["channel"] == channel].copy()
        cost = float(CHANNEL_PARAMS[channel]["hourly_cost"])
        shift_frames.append(optimise_channel(channel, subset, cost))
        print(f"Solved {channel}")

    optimal = pd.concat(shift_frames, ignore_index=True)
    optimal_path = DATA / "optimal_shifts.csv"
    optimal.to_csv(optimal_path, index=False)

    rows = []
    for channel in CHANNELS:
        base_cov = coverage_by_dow(baseline, channel)
        opt_cov = coverage_by_dow(optimal, channel)
        subset = requirements[requirements["channel"] == channel]
        for _, req in subset.iterrows():
            dow = int(req["dow"])
            hour = int(req["hour"])
            base_rostered = int(base_cov[dow, hour])
            opt_rostered = int(opt_cov[dow, hour])
            rows.append(
                {
                    "channel": channel,
                    "dow": dow,
                    "dow_name": DOW_NAMES[dow],
                    "hour": hour,
                    "interval_key": f"{channel}-{dow}-{hour}",
                    "method": req["method"],
                    "avg_arrivals": req["avg_arrivals"],
                    "aht_sec": req["aht_sec"],
                    "erlangs": req["erlangs"],
                    "sla_sec": req["sla_sec"],
                    "sla_pct": req["sla_pct"],
                    "logged_in_required": int(req["logged_in_required"]),
                    "rostered_required": int(req["rostered_required"]),
                    "predicted_sl": req["predicted_sl"],
                    "predicted_asa_sec": req["predicted_asa_sec"],
                    "predicted_occupancy": req["occupancy"],
                    "baseline_rostered": base_rostered,
                    "optimal_rostered": opt_rostered,
                    "baseline_logged_in": logged_in_from_rostered(base_rostered),
                    "optimal_logged_in": logged_in_from_rostered(opt_rostered),
                    "hourly_cost": float(CHANNEL_PARAMS[channel]["hourly_cost"]),
                }
            )

    coverage = pd.DataFrame(rows)
    coverage_path = DATA / "coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    cost_rows = []
    for channel in CHANNELS:
        cost = float(CHANNEL_PARAMS[channel]["hourly_cost"])
        base_hours = float(baseline.loc[baseline["channel"] == channel, "agents"].sum()) * SHIFT_HOURS
        opt_hours = float(optimal.loc[optimal["channel"] == channel, "agents"].sum()) * SHIFT_HOURS
        base_cost = base_hours * cost
        opt_cost = opt_hours * cost
        gap = coverage[coverage["channel"] == channel]
        short_hours = int((gap["baseline_logged_in"] < gap["logged_in_required"]).sum())
        cost_rows.append(
            {
                "channel": channel,
                "baseline_agent_hours": base_hours,
                "optimal_agent_hours": opt_hours,
                "hourly_cost": cost,
                "baseline_weekly_cost": round(base_cost, 2),
                "optimal_weekly_cost": round(opt_cost, 2),
                "weekly_savings": round(base_cost - opt_cost, 2),
                "savings_pct": round(100.0 * (base_cost - opt_cost) / base_cost, 1) if base_cost else 0.0,
                "baseline_hours_below_requirement": short_hours,
            }
        )
    costs = pd.DataFrame(cost_rows)
    costs_path = DATA / "cost_summary.csv"
    costs.to_csv(costs_path, index=False)
    _plot_monday(coverage)

    print(costs.to_string(index=False))
    print(f"Wrote {optimal_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {costs_path}")
    print(f"Wrote {DATA / 'plots' / 'phone_monday_coverage.png'}")


if __name__ == "__main__":
    main()
