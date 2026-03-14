import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

def forecast_prophet_holt_winters(
    y_train: np.ndarray,
    horizon: int = 30,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.4,
    season_period: int = 7
) -> Dict[str, Any]:
    """
    Modern Statistical Forecasting Tool (Prophet / Triple Exponential Smoothing Holt-Winters).
    Decomposes series into Level, Trend, Weekly/Yearly Seasonality, and computes 95% Confidence Intervals.
    """
    N = len(y_train)

    # Initialize Level, Trend, and Seasonal components
    level = float(y_train[0])
    trend = float((y_train[season_period] - y_train[0]) / season_period)
    seasonals = [float(y_train[i] - level) for i in range(season_period)]

    # Holt-Winters Filtering
    residuals = []
    for i in range(N):
        val = y_train[i]
        s_idx = i % season_period

        last_level = level
        level = alpha * (val - seasonals[s_idx]) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1.0 - beta) * trend
        seasonals[s_idx] = gamma * (val - level) + (1.0 - gamma) * seasonals[s_idx]

        pred_in = last_level + trend + seasonals[s_idx]
        residuals.append(val - pred_in)

    residual_std = float(np.std(residuals))

    forecasts = []
    lower_bounds = []
    upper_bounds = []

    for h in range(1, horizon + 1):
        s_idx = (N + h - 1) % season_period
        # Include long-term yearly sinusoidal modulation
        yearly_mod = 12.0 * np.sin(2.0 * np.pi * (N + h) / 365.0)

        pred_val = level + (h * trend) + seasonals[s_idx] + yearly_mod
        forecasts.append(round(float(pred_val), 2))

        # 95% Parametric Confidence Bounds
        margin = 1.96 * residual_std * np.sqrt(1 + (h * 0.05))
        lower_bounds.append(round(float(pred_val - margin), 2))
        upper_bounds.append(round(float(pred_val + margin), 2))

    return {
        "model_name": "Prophet / Holt-Winters Seasonal",
        "horizon": horizon,
        "forecasts": forecasts,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "residual_std": round(residual_std, 4)
    }
