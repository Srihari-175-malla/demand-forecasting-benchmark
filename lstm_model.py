import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any

class PyTorchLSTMForecaster(nn.Module):
    """
    2-Layer PyTorch LSTM Deep Learning Model for Time Series Forecasting.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(PyTorchLSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        out, _ = self.lstm(x, (h0, c0))
        # Use last time step output
        last_out = out[:, -1, :]
        return self.fc(last_out)


def forecast_lstm_pytorch(
    y_train: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
    epochs: int = 40,
    lr: float = 0.005
) -> Dict[str, Any]:
    """
    Train PyTorch LSTM model on sliding windows and generate multi-step forecasts with 95% confidence bounds.
    """
    # 1. MinMax Normalization
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_scaled = (y_train - y_min) / (y_max - y_min + 1e-12)

    # 2. Build Sliding Window Dataset (B, L, 1) -> Target y_{t+1}
    X_wins = []
    Y_wins = []
    for i in range(len(y_scaled) - lookback):
        X_wins.append(y_scaled[i:i+lookback])
        Y_wins.append(y_scaled[i+lookback])

    X_tensor = torch.tensor(np.array(X_wins), dtype=torch.float32).unsqueeze(-1)
    Y_tensor = torch.tensor(np.array(Y_wins), dtype=torch.float32).unsqueeze(-1)

    # 3. Model Training
    model = PyTorchLSTMForecaster(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, Y_tensor)
        loss.backward()
        optimizer.step()

    # 4. Autoregressive Multi-Step Forecast
    model.eval()
    curr_win = list(y_scaled[-lookback:])
    scaled_forecasts = []

    with torch.no_grad():
        for _ in range(horizon):
            inp = torch.tensor(np.array(curr_win[-lookback:]), dtype=torch.float32).view(1, lookback, 1)
            pred_s = float(model(inp).item())
            scaled_forecasts.append(pred_s)
            curr_win.append(pred_s)

    # Inverse Scale
    forecasts = [round(float(s * (y_max - y_min) + y_min), 2) for s in scaled_forecasts]

    # Compute Residual Uncertainty for 95% Confidence Bounds
    model_train_preds = model(X_tensor).squeeze(-1).detach().numpy() * (y_max - y_min) + y_min
    train_targets = y_train[lookback:]
    residuals = train_targets - model_train_preds
    residual_std = float(np.std(residuals))

    lower_bounds = []
    upper_bounds = []
    for h, f_val in enumerate(forecasts, start=1):
        margin = 1.96 * residual_std * np.sqrt(1 + (h * 0.08))
        lower_bounds.append(round(float(f_val - margin), 2))
        upper_bounds.append(round(float(f_val + margin), 2))

    return {
        "model_name": "PyTorch LSTM Deep Learning",
        "horizon": horizon,
        "forecasts": forecasts,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "residual_std": round(residual_std, 4)
    }
