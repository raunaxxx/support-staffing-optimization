import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

DATA = Path("data")
DATA.mkdir(exist_ok=True)

channels = {
    "phone": {"aht": 360},
    "chat": {"aht": 480},
    "email": {"aht": 540}
}

rows = []

start = pd.Timestamp("2026-01-05")

contact_id = 1

for day in range(28):  # 4 weeks

    current_day = start + pd.Timedelta(days=day)

    for hour in range(24):

        base_volume = max(
            2,
            int(
                20
                + 40 * np.sin((hour - 8) / 24 * np.pi)
            )
        )

        for channel in channels:

            n = rng.poisson(base_volume)

            for _ in range(n):

                arrival = (
                    current_day
                    + pd.Timedelta(hours=hour)
                    + pd.to_timedelta(
                        rng.uniform(0, 3600),
                        unit="s"
                    )
                )

                aht = rng.lognormal(
                    mean=np.log(channels[channel]["aht"]),
                    sigma=0.4
                )

                abandoned = rng.choice(
                    [0, 1],
                    p=[0.95, 0.05]
                )

                rows.append([
                    contact_id,
                    arrival,
                    channel,
                    round(aht, 0),
                    abandoned
                ])

                contact_id += 1

contacts = pd.DataFrame(
    rows,
    columns=[
        "contact_id",
        "arrived_at",
        "channel",
        "handle_time_sec",
        "abandoned"
    ]
)

contacts.to_csv(
    DATA / "contacts.csv",
    index=False
)

print(
    f"Generated {len(contacts):,} contacts"
)
