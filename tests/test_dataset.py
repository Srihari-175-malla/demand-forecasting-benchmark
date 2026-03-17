import unittest
from forecasting_study.dataset import generate_daily_energy_demand_dataset

class TestDataset(unittest.TestCase):
    def test_daily_energy_dataset_generation(self):
        ds = generate_daily_energy_demand_dataset(n_days=730, seed=42)
        self.assertEqual(ds["n_days"], 730)
        self.assertEqual(len(ds["y_full"]), 730)
        self.assertEqual(len(ds["y_train"]), 584)
        self.assertEqual(len(ds["y_test"]), 146)

if __name__ == "__main__":
    unittest.main()
