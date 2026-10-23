"""
Generate the simulated minute-level ENMO sample shipped with CosinorAge.

The output (``sample_enmo.csv``) contains no participant data. It is drawn
from a simple rest-activity model: a cosine-shaped daytime activity level,
a low-activity sleep window, gamma-distributed minute-to-minute noise and a
few short bouts of moderate-to-vigorous activity per day.

The file uses the Samsung Galaxy Watch CSV layout expected by
``GalaxyDataHandler(data_format='csv')``: a ``time`` column with ISO 8601
UTC timestamps and an ``enmo_mg`` column with ENMO values in milli-g.

Run from the repository root to regenerate the file:

    python examples/data/generate_sample_data.py
"""

import os

import numpy as np
import pandas as pd

SEED = 42
START = "2024-01-01 12:00:00"
N_DAYS = 8  # starts and ends at noon, so 7 complete days remain after filtering
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_enmo.csv")


def simulate_enmo(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range(START, periods=N_DAYS * 1440, freq="1min")
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60

    enmo = np.empty(len(time))
    for day in np.unique(time.normalize()):
        idx = np.where(time.normalize() == day)[0]
        h = hour[idx]

        # day-specific sleep timing and activity level
        wake = 7.0 + rng.normal(0, 0.5)
        sleep = 23.0 + rng.normal(0, 0.5)
        level = rng.uniform(25, 40)
        awake = (h >= wake) & (h < sleep)

        # cosine-shaped daytime activity peaking mid-afternoon
        mean = np.where(
            awake,
            level * (1 + 0.6 * np.cos(2 * np.pi * (h - 15) / 24)),
            3.0,
        )
        values = rng.gamma(shape=2.0, scale=mean / 2.0)

        # a few 10-30 minute bouts of moderate-to-vigorous activity
        awake_idx = np.where(awake)[0]
        for _ in range(rng.integers(1, 4)):
            start = rng.choice(awake_idx[:-30])
            length = rng.integers(10, 31)
            values[start:start + length] += rng.uniform(100, 250)

        enmo[idx] = values

    return pd.DataFrame(
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "enmo_mg": np.round(enmo, 3),
        }
    )


if __name__ == "__main__":
    data = simulate_enmo()
    data.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(data)} minutes of simulated ENMO data to {OUTPUT}")
