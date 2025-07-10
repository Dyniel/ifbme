import unittest
import pandas as pd
import numpy as np

from features.trend_features import make_trends # Adjust import path

class TestTrendFeatures(unittest.TestCase):

    def setUp(self):
        # Basic data for testing
        self.num_patients = 2
        self.records_per_patient = 5
        self.data = {
            'patient_id': np.repeat(np.arange(self.num_patients), self.records_per_patient),
            'hour': np.tile(np.arange(self.records_per_patient), self.num_patients),
            'heart_rate': [60, 62, 61, 63, 62,  70, 72, np.nan, 73, 71], # Patient 0, Patient 1 (with NaN)
            'temperature': [37.0, 37.1, 37.0, 36.9, 37.2, 38.0, 38.1, 38.2, 38.0, 37.9]
        }
        self.df = pd.DataFrame(self.data)
        # Ensure sorted as make_trends expects or handles this
        self.df = self.df.sort_values(by=['patient_id', 'hour']).reset_index(drop=True)

        self.id_col = 'patient_id'
        self.time_col = 'hour'
        self.value_cols = ['heart_rate', 'temperature']
        self.windows = [3] # Test with a single window for simplicity

    def test_output_columns_exist(self):
        """Test if the function creates the expected new columns."""
        df_out = make_trends(self.df.copy(), self.id_col, self.time_col, self.value_cols, self.windows)

        for val_col in self.value_cols:
            for win in self.windows:
                self.assertIn(f'{val_col}_mean_{win}h', df_out.columns)
                self.assertIn(f'{val_col}_slope_{win}h', df_out.columns)
                self.assertIn(f'{val_col}_var_{win}h', df_out.columns)

    def test_rolling_mean_calculation(self):
        """Test a specific rolling mean calculation."""
        df_out = make_trends(self.df.copy(), self.id_col, self.time_col, ['heart_rate'], windows=[3])

        # Patient 0, heart_rate: [60, 62, 61, 63, 62]
        # Expected means for window=3:
        # idx 0 (val 60): mean(60) = 60
        # idx 1 (val 62): mean(60,62) = 61
        # idx 2 (val 61): mean(60,62,61) = 61
        # idx 3 (val 63): mean(62,61,63) = 62
        # idx 4 (val 62): mean(61,63,62) = 62
        expected_means_p0 = [60.0, 61.0, 61.0, 62.0, 62.0]
        # Patient 1, heart_rate: [70, 72, np.nan, 73, 71]
        # Expected means for window=3 (min_periods=1 for mean):
        # idx 5 (val 70): mean(70) = 70
        # idx 6 (val 72): mean(70,72) = 71
        # idx 7 (val nan):mean(70,72,nan) -> mean(70,72) = 71 (if nan is ignored by rolling.mean by default)
        # idx 8 (val 73): mean(72,nan,73) -> mean(72,73) = 72.5
        # idx 9 (val 71): mean(nan,73,71) -> mean(73,71) = 72
        # Pandas rolling mean default skipna=True
        expected_means_p1 = [70.0, 71.0, 71.0, 72.5, 72.0]

        self.assertTrue(np.allclose(df_out.loc[df_out['patient_id']==0, 'heart_rate_mean_3h'].values, expected_means_p0))
        self.assertTrue(np.allclose(df_out.loc[df_out['patient_id']==1, 'heart_rate_mean_3h'].values, expected_means_p1))

    def test_rolling_std_dev_calculation(self):
        """Test rolling standard deviation (variability)."""
        df_out = make_trends(self.df.copy(), self.id_col, self.time_col, ['heart_rate'], windows=[3])
        # Patient 0, heart_rate: [60, 62, 61, 63, 62]
        # Expected std for window=3, min_periods=1 (fillna(0) for single point std=nan):
        # idx 0 (val 60): std(60) = 0
        # idx 1 (val 62): std(60,62) = 1.41421356
        # idx 2 (val 61): std(60,62,61) = 1.0
        # idx 3 (val 63): std(62,61,63) = 1.0
        # idx 4 (val 62): std(61,63,62) = 1.0
        expected_std_p0 = [0.0, np.std([60,62], ddof=0), np.std([60,62,61], ddof=0), np.std([62,61,63], ddof=0), np.std([61,63,62], ddof=0)]
        # Note: pandas .std() default ddof=1. For population std (ddof=0) as in np.std default:
        # Need to be careful about which std is used. Pandas default is sample std.
        # Let's use pandas' calculation for direct comparison.
        p0_hr = pd.Series([60,62,61,63,62])
        expected_pd_std_p0 = [0.0] + [p0_hr.iloc[:i+1].rolling(window=3,min_periods=1).std().iloc[-1] for i in range(len(p0_hr)-1)]
        expected_pd_std_p0 = p0_hr.rolling(window=3,min_periods=1).std().fillna(0).tolist()

        self.assertTrue(np.allclose(df_out.loc[df_out['patient_id']==0, 'heart_rate_var_3h'].values, expected_pd_std_p0))


    def test_slope_calculation_basic(self):
        """Test a basic slope calculation."""
        # For Patient 0, heart_rate: [60, 62, 61, 63, 62], hours: [0, 1, 2, 3, 4]
        # Window = 3, min_periods_slope = 2
        # idx 0: nan (not enough points for slope with window 3, min_periods=2) -> filled with 0 by make_trends
        # idx 1: slope([60,62] vs [0,1]) = (62-60)/(1-0) = 2.0
        # idx 2: slope([60,62,61] vs [0,1,2]) -> linregress([0,1,2], [60,62,61]).slope = 0.5
        # idx 3: slope([62,61,63] vs [1,2,3]) -> linregress([1,2,3], [62,61,63]).slope = 0.5
        # idx 4: slope([61,63,62] vs [2,3,4]) -> linregress([2,3,4], [61,63,62]).slope = 0.5

        from scipy.stats import linregress
        expected_slopes_p0 = [
            0.0, # Not enough for window=3 for slope, then filled
            linregress([0,1], [60,62]).slope if len([60,62]) >=2 else 0.0, # This is rolling(2).apply essentially.
                                                                        # Our rolling(3).apply will see [60,62] at hour 1 due to window size.
            linregress(self.df.loc[0:2,'hour'], self.df.loc[0:2,'heart_rate']).slope, # window [0,1,2] for point 2
            linregress(self.df.loc[1:3,'hour'], self.df.loc[1:3,'heart_rate']).slope, # window [1,2,3] for point 3
            linregress(self.df.loc[2:4,'hour'], self.df.loc[2:4,'heart_rate']).slope  # window [2,3,4] for point 4
        ]
        # The current make_trends slope applies rolling(window).apply, so for idx 1 (hour 1), window=3 means it looks at data[0:2]
        # The first valid slope calculation (min_periods=2) will be at index 1.
        # For hour 0 (idx 0), rolling(3).apply with min_periods=2 on series [60] -> nan -> 0
        # For hour 1 (idx 1), rolling(3).apply with min_periods=2 on series [60,62] -> slope( (0,60), (1,62) ) = 2.0
        # For hour 2 (idx 2), rolling(3).apply with min_periods=2 on series [60,62,61] -> slope( (0,60),(1,62),(2,61) ) = 0.5

        df_out = make_trends(self.df.copy(), self.id_col, self.time_col, ['heart_rate'], windows=[3], slope_window_min_periods=2)
        actual_slopes_p0 = df_out.loc[df_out['patient_id']==0, 'heart_rate_slope_3h'].values

        # Recompute expected based on how apply works with rolling
        calc_expected_slopes_p0 = [0.0] # For hour 0, not enough data in window
        series_p0_hr = self.df.loc[self.df['patient_id']==0, 'heart_rate']
        series_p0_h = self.df.loc[self.df['patient_id']==0, 'hour']

        if len(series_p0_hr[:2]) >= 2: calc_expected_slopes_p0.append(linregress(series_p0_h[:2], series_p0_hr[:2]).slope)
        else: calc_expected_slopes_p0.append(0.0)

        if len(series_p0_hr[:3]) >= 2: calc_expected_slopes_p0.append(linregress(series_p0_h[:3], series_p0_hr[:3]).slope)
        else: calc_expected_slopes_p0.append(0.0)

        if len(series_p0_hr[1:4]) >= 2: calc_expected_slopes_p0.append(linregress(series_p0_h[1:4], series_p0_hr[1:4]).slope) # for hour 3, window is [1,2,3]
        else: calc_expected_slopes_p0.append(0.0)

        if len(series_p0_hr[2:5]) >= 2: calc_expected_slopes_p0.append(linregress(series_p0_h[2:5], series_p0_hr[2:5]).slope) # for hour 4, window is [2,3,4]
        else: calc_expected_slopes_p0.append(0.0)

        self.assertTrue(np.allclose(actual_slopes_p0, calc_expected_slopes_p0))


    def test_handle_nans_in_slope(self):
        """Test slope calculation when NaNs are present."""
        df_out = make_trends(self.df.copy(), self.id_col, self.time_col, ['heart_rate'], windows=[3], slope_window_min_periods=2)
        # Patient 1, heart_rate: [70, 72, np.nan, 73, 71], hours: [0, 1, 2, 3, 4] (relative to patient 1 start)
        # hour_abs: [5,6,7,8,9]
        # idx 5 (val 70, abs_hr 0 for p1): slope on [70] -> nan -> 0
        # idx 6 (val 72, abs_hr 1 for p1): slope on [70,72] vs [0,1] (relative time) -> 2.0
        # idx 7 (val nan,abs_hr 2 for p1): slope on [70,72,nan] vs [0,1,2] -> effectively [70,72] vs [0,1] -> 2.0
        # idx 8 (val 73, abs_hr 3 for p1): slope on [72,nan,73] vs [1,2,3] -> effectively [72,73] vs [1,3] (if time is used from original df) -> (73-72)/(3-1) = 0.5
        # idx 9 (val 71, abs_hr 4 for p1): slope on [nan,73,71] vs [2,3,4] -> effectively [73,71] vs [3,4] -> (71-73)/(4-3) = -2.0

        actual_slopes_p1 = df_out.loc[df_out['patient_id']==1, 'heart_rate_slope_3h'].values

        from scipy.stats import linregress
        hr_p1 = self.df.loc[self.df['patient_id']==1, 'heart_rate'].values # [70, 72, nan, 73, 71]
        h_p1_orig = self.df.loc[self.df['patient_id']==1, 'hour'].values # [0,1,2,3,4] (relative index for this patient)

        expected_slopes_p1 = []
        # Hour 0 (p1)
        expected_slopes_p1.append(0.0)
        # Hour 1 (p1)
        series_vals = hr_p1[0:2]; series_h = h_p1_orig[0:2]
        valid_idx = ~np.isnan(series_vals)
        if np.sum(valid_idx) >= 2: expected_slopes_p1.append(linregress(series_h[valid_idx], series_vals[valid_idx]).slope)
        else: expected_slopes_p1.append(0.0)
        # Hour 2 (p1)
        series_vals = hr_p1[0:3]; series_h = h_p1_orig[0:3]
        valid_idx = ~np.isnan(series_vals)
        if np.sum(valid_idx) >= 2: expected_slopes_p1.append(linregress(series_h[valid_idx], series_vals[valid_idx]).slope)
        else: expected_slopes_p1.append(0.0)
        # Hour 3 (p1)
        series_vals = hr_p1[1:4]; series_h = h_p1_orig[1:4] # Window for current point at index 3 is [1,2,3]
        valid_idx = ~np.isnan(series_vals)
        if np.sum(valid_idx) >= 2: expected_slopes_p1.append(linregress(series_h[valid_idx], series_vals[valid_idx]).slope)
        else: expected_slopes_p1.append(0.0)
        # Hour 4 (p1)
        series_vals = hr_p1[2:5]; series_h = h_p1_orig[2:5] # Window for current point at index 4 is [2,3,4]
        valid_idx = ~np.isnan(series_vals)
        if np.sum(valid_idx) >= 2: expected_slopes_p1.append(linregress(series_h[valid_idx], series_vals[valid_idx]).slope)
        else: expected_slopes_p1.append(0.0)

        self.assertTrue(np.allclose(actual_slopes_p1, expected_slopes_p1))

    def test_empty_df(self):
        """Test with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['patient_id', 'hour', 'heart_rate'])
        df_out = make_trends(empty_df, self.id_col, self.time_col, ['heart_rate'], self.windows)
        self.assertTrue(df_out.empty) # Should return an empty df with new columns if structure allows, or just empty
        # Current make_trends will add columns even if empty, so check columns
        for val_col in ['heart_rate']:
            for win in self.windows:
                self.assertIn(f'{val_col}_mean_{win}h', df_out.columns)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
