import unittest
import numpy as np
from forecasting_study.dataset import generate_daily_energy_demand_dataset
from forecasting_study.sarima_model import run_adf_stationarity_test, compute_acf_pacf, forecast_sarima
from forecasting_study.prophet_model import forecast_prophet_holt_winters
from forecasting_study.lstm_model import forecast_lstm_pytorch
from forecasting_study.evaluation import calculate_metrics, generate_comparative_tradeoff_analysis

class TestModelsAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.ds = generate_daily_energy_demand_dataset(n_days=300, seed=42)

    def test_adf_and_sarima(self):
        adf_res = run_adf_stationarity_test(self.ds["y_train"])
        self.assertIn("adf_statistic", adf_res)

        sarima_res = forecast_sarima(self.ds["y_train"], horizon=7)
        self.assertEqual(len(sarima_res["forecasts"]), 7)
        self.assertEqual(len(sarima_res["lower_bounds"]), 7)

    def test_prophet_and_lstm(self):
        prophet_res = forecast_prophet_holt_winters(self.ds["y_train"], horizon=7)
        self.assertEqual(len(prophet_res["forecasts"]), 7)

        lstm_res = forecast_lstm_pytorch(self.ds["y_train"], horizon=7, epochs=10)
        self.assertEqual(len(lstm_res["forecasts"]), 7)

    def test_metrics_evaluation(self):
        y_true = np.array([100.0, 105.0, 110.0])
        y_pred = np.array([102.0, 104.0, 108.0])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertGreater(metrics["mae"], 0.0)
        self.assertGreater(metrics["rmse"], 0.0)

if __name__ == "__main__":
    unittest.main()
