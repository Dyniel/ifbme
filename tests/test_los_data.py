import unittest
import pandas as pd
from data_utils.data_loader import load_raw_data
from configs.dummy_train_config import get_dummy_config

class TestLoSData(unittest.TestCase):

    def test_los_data_is_valid(self):
        """
        This test checks if the lengthofStay column in the training data is valid.
        A valid LoS is a non-negative number.
        """
        config = get_dummy_config()
        X_full_raw_df, y_full_raw_series = load_raw_data(config, base_data_path=config.get('data_dir', 'data/'))

        los_column_name = config.get('los_column', 'lengthofStay')
        self.assertTrue(los_column_name in X_full_raw_df.columns, f"'{los_column_name}' not found in data columns")

        los_data = X_full_raw_df[los_column_name]

        # Check for non-numeric values
        non_numeric = los_data[~pd.to_numeric(los_data, errors='coerce').notna()]
        self.assertEqual(len(non_numeric), 0, f"Found non-numeric LoS values: {non_numeric.tolist()}")

        # Check for negative values
        los_data_numeric = pd.to_numeric(los_data)
        negative_values = los_data_numeric[los_data_numeric < 0]
        self.assertEqual(len(negative_values), 0, f"Found negative LoS values: {negative_values.tolist()}")

        # Check for NaN values
        nan_values = los_data_numeric.isna().sum()
        print(f"Found {nan_values} NaN LoS values out of {len(los_data)} total samples.")

if __name__ == '__main__':
    unittest.main()
