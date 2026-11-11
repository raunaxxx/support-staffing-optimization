import pandas as pd
from pathlib import Path

data = Path("data")

req = pd.read_csv(data / "staffing_requirements.csv")

baseline = req.copy()

baseline["start_hour"] = baseline["hour"]
baseline["agents"] = baseline["rostered_required"]

baseline[
    [
        "channel",
        "dow",
        "start_hour",
        "agents"
    ]
].to_csv(
    data / "baseline_shifts.csv",
    index=False
)

print("Created baseline_shifts.csv")
