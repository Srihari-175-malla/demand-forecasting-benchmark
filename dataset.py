import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

def generate_daily_energy_demand_dataset(n_days: int = 730, seed: int = 42) -> Dict[str, Any]:
    """
    Generate 2 years (730 days) of daily energy demand (kW) with trend, weekly/yearly seasonality, and noise.
    """
    np.random.seed(seed)
    t = np.arange(n_days)

    # 1. Base Load + Trend
    base_load = 500.0
    trend = 0.25 * t

    # 2. Weekly Seasonality (7-day pattern: higher on Mon-Fri, lower on Sat-Sun)
    day_of_week = t % 7
    weekly_seasonality = np.where(day_of_week < 5, 45.0, -35.0)

    # 3. Yearly Seasonality (365-day sinusoidal pattern for summer/winter peak)
    yearly_seasonality = 65.0 * np.sin(2.0 * np.pi * t / 365.0 - 0.5)

    # 4. Non-linear noise
    noise = np.random.normal(loc=0.0, scale=12.0, size=n_days)

    y_full = base_load + trend + weekly_seasonality + yearly_seasonality + noise
    y_full = np.round(y_full, 2)

    # Train / Test split (80% train = 584 days, 20% test = 146 days)
    split_idx = int(0.80 * n_days)
    y_train = y_full[:split_idx]
    y_test = y_full[split_idx:]

    date_range = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    date_labels = [d.strftime("%Y-%m-%d") for d in date_range]

    return {
        "n_days": n_days,
        "split_idx": split_idx,
        "date_labels": date_labels,
        "y_full": y_full,
        "y_train": y_train,
        "y_test": y_test,
        "train_dates": date_labels[:split_idx],
        "test_dates": date_labels[split_idx:]
    }
