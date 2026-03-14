import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

from .dataset import generate_daily_energy_demand_dataset
from .sarima_model import run_adf_stationarity_test, compute_acf_pacf, forecast_sarima
from .prophet_model import forecast_prophet_holt_winters
from .lstm_model import forecast_lstm_pytorch
from .evaluation import calculate_metrics, generate_comparative_tradeoff_analysis

app = FastAPI(title="Comparative Time Series Forecasting Study Platform", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global Dataset State
dataset_db = generate_daily_energy_demand_dataset(n_days=730, seed=42)

class ForecastRequest(BaseModel):
    horizon: int = 7 # 7 for Short-Term, 30 for Long-Term

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/stationarity")
async def api_stationarity():
    y_train = dataset_db["y_train"]
    adf_res = run_adf_stationarity_test(y_train)
    acf_pacf_res = compute_acf_pacf(y_train, max_lags=14)

    return {
        "adf_test": adf_res,
        "acf_pacf": acf_pacf_res
    }

@app.post("/api/forecast")
async def api_forecast(req: ForecastRequest):
    horizon = req.horizon
    y_train = dataset_db["y_train"]
    y_test = dataset_db["y_test"][:horizon]
    test_dates = dataset_db["test_dates"][:horizon]

    # 1. Classical SARIMA Model
    sarima_res = forecast_sarima(y_train, horizon=horizon)
    sarima_metrics = calculate_metrics(y_test, sarima_res["forecasts"])
    sarima_res["metrics"] = sarima_metrics

    # 2. Modern Prophet / Holt-Winters Model
    prophet_res = forecast_prophet_holt_winters(y_train, horizon=horizon)
    prophet_metrics = calculate_metrics(y_test, prophet_res["forecasts"])
    prophet_res["metrics"] = prophet_metrics

    # 3. PyTorch LSTM Deep Learning Model
    lstm_res = forecast_lstm_pytorch(y_train, horizon=horizon, epochs=30)
    lstm_metrics = calculate_metrics(y_test, lstm_res["forecasts"])
    lstm_res["metrics"] = lstm_metrics

    # Comparative Metric Rankings
    evaluations = [
        {"model_name": "SARIMA", **sarima_metrics},
        {"model_name": "Prophet / Holt-Winters", **prophet_metrics},
        {"model_name": "PyTorch LSTM", **lstm_metrics}
    ]

    tradeoff_analysis = generate_comparative_tradeoff_analysis(evaluations, evaluations)

    return {
        "horizon": horizon,
        "test_dates": test_dates,
        "y_true": y_test.tolist(),
        "models": {
            "sarima": sarima_res,
            "prophet": prophet_res,
            "lstm": lstm_res
        },
        "evaluation_rankings": evaluations,
        "tradeoff_analysis": tradeoff_analysis
    }
