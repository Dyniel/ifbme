import pandas as pd
import numpy as np
from scipy.stats import linregress

def make_trends(df, id_col, time_col, value_cols, windows=[1, 3, 12], slope_window_min_periods=2):
    """
    Generates trend features (rolling mean, slope, variability) for specified value columns.

    Args:
        df (pd.DataFrame): Input DataFrame. Must be sorted by id_col and time_col.
        id_col (str): Column name for patient or subject ID.
        time_col (str): Column name for time (e.g., hours from admission).
                        Assumed to be numeric and equally spaced for slope calculation,
                        or slope calculation might be less accurate.
        value_cols (list): List of column names for which to calculate trend features
                           (e.g., ['heart_rate', 'temperature']).
        windows (list): List of window sizes (in terms of number of records) for rolling calculations.
        slope_window_min_periods (int): Minimum number of periods required to calculate slope.
                                        Useful to avoid slopes from too few points.

    Returns:
        pd.DataFrame: DataFrame with new trend features appended.
                      New columns will be named e.g., '{value_col}_mean_{window}h',
                      '{value_col}_slope_{window}h', '{value_col}_var_{window}h'.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in [id_col, time_col] + value_cols):
        raise ValueError("One or more specified columns not in DataFrame.")
    if not df[time_col].is_monotonic_increasing and not df.groupby(id_col)[time_col].is_monotonic_increasing.all():
         print(f"Warning: DataFrame time column '{time_col}' is not globally sorted. "
               "Ensure it is sorted within each group defined by '{id_col}' for correct window operations.")


    # It's crucial that data is sorted by ID and then by time for rolling operations per ID.
    # df = df.sort_values(by=[id_col, time_col]) # Ensure sorting, or expect user to provide sorted data

    # Store new feature columns to concatenate later
    all_new_features = []

    for value_col in value_cols:
        for window in windows:
            # Group by ID
            grouped = df.groupby(id_col)[value_col]

            # --- Rolling Mean ---
            mean_col_name = f'{value_col}_mean_{window}h'
            # min_periods=1 means if there's at least one observation in window, calculate mean
            df[mean_col_name] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # --- Rolling Variability (Standard Deviation) ---
            var_col_name = f'{value_col}_var_{window}h'
            # min_periods=1, if std of 1 point is nan, then it's fine.
            df[var_col_name] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            # Rolling std of a single point is NaN. Fill with 0 as variability of one point is zero.
            df[var_col_name] = df[var_col_name].fillna(0)


            # --- Rolling Slope (Linear Regression Slope) ---
            slope_col_name = f'{value_col}_slope_{window}h'

            # Helper function to apply to each group's rolling window
            def calculate_slope(series):
                # `series` here is a segment of `value_col` for a specific ID and window
                # We need corresponding time values for regression
                # This requires access to the time_col for the same window.
                # Using an index that aligns with the original df for time lookup.

                time_values = df.loc[series.index, time_col] # Get corresponding time values

                if len(series) < slope_window_min_periods or series.isnull().sum() > len(series) - slope_window_min_periods :
                    return np.nan # Not enough data points to calculate a reliable slope

                # Drop NaNs for regression
                valid_indices = ~series.isnull()
                current_values = series[valid_indices]
                current_times = time_values[valid_indices]

                if len(current_values) < slope_window_min_periods:
                    return np.nan

                # If all y values are the same, slope is 0 (linregress might return NaN or error)
                if len(np.unique(current_values)) == 1:
                    return 0.0

                try:
                    # Perform linear regression: value_col ~ time_col
                    slope, _, _, _, _ = linregress(current_times, current_values)
                    return slope
                except ValueError: # Catches issues if linregress fails
                    return np.nan

            # The .rolling().apply() sends a Series for each window to the function.
            # We need to ensure 'raw=False' so it passes a Series with an index.
            # This is computationally intensive.
            df[slope_col_name] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=slope_window_min_periods).apply(calculate_slope, raw=False)
            )
            # Fill NaNs for slopes if any (e.g. for initial periods or where slope couldn't be calculated)
            # A common strategy is to fill with 0, assuming no trend if not calculable.
            df[slope_col_name] = df[slope_col_name].fillna(0)


    return df

if __name__ == '__main__':
    # Create a sample DataFrame
    num_patients = 3
    records_per_patient = 20
    data = {
        'patient_id': np.repeat(np.arange(num_patients), records_per_patient),
        'hour': np.tile(np.arange(records_per_patient), num_patients),
        'heart_rate': np.random.randint(60, 100, num_patients * records_per_patient) + \
                      np.repeat(np.random.randn(num_patients) * 5, records_per_patient) + \
                      np.tile(np.arange(records_per_patient), num_patients) * np.random.choice([-0.5, 0.5, 0.2, -0.3], num_patients * records_per_patient).reshape(num_patients,records_per_patient).flatten() , # Add some trend
        'temperature': np.random.normal(37, 0.5, num_patients * records_per_patient) + \
                       np.repeat(np.random.randn(num_patients) * 0.2, records_per_patient)
    }
    sample_df = pd.DataFrame(data)
    sample_df.loc[sample_df['patient_id'] == 1, 'heart_rate'] = sample_df.loc[sample_df['patient_id'] == 1, 'heart_rate'] + np.arange(records_per_patient) * 1.5
    sample_df.loc[5:10, 'heart_rate'] = np.nan # Add some NaNs
    sample_df.loc[25:28, 'temperature'] = np.nan


    print("Original DataFrame:")
    print(sample_df.head(25))

    # Ensure DataFrame is sorted by patient_id and hour for correct processing
    sample_df_sorted = sample_df.sort_values(by=['patient_id', 'hour']).reset_index(drop=True)

    # Define parameters for trend feature generation
    id_column = 'patient_id'
    time_column = 'hour'
    value_columns = ['heart_rate', 'temperature']
    window_sizes = [3, 6, 12] # Example window sizes (in hours/records)

    # Generate trend features
    df_with_trends = make_trends(
        sample_df_sorted.copy(), # Use a copy to keep original df unchanged
        id_col=id_column,
        time_col=time_column,
        value_cols=value_columns,
        windows=window_sizes,
        slope_window_min_periods=2 # Require at least 2 points for slope
    )

    print("\nDataFrame with Trend Features:")
    pd.set_option('display.max_columns', None) # Show all columns
    print(df_with_trends.head(25))

    # Check for one patient
    print("\nTrend features for patient_id = 1:")
    print(df_with_trends[df_with_trends['patient_id'] == 1].tail(15))

    # Test with a single value column and single window
    df_single_trend = make_trends(
        sample_df_sorted.copy(),
        id_col=id_column,
        time_col=time_column,
        value_cols=['heart_rate'],
        windows=[6]
    )
    print("\nDataFrame with single trend (heart_rate, window 6):")
    print(df_single_trend[df_single_trend['patient_id'] == 0].head(10))

    # Test edge case: not enough data for slope
    short_df_data = {
        'patient_id': [0,0,0,1,1],
        'hour': [0,1,2,0,1],
        'value': [10,12,11,20,22]
    }
    short_df = pd.DataFrame(short_df_data)
    short_df_trends = make_trends(short_df, 'patient_id', 'hour', ['value'], windows=[3], slope_window_min_periods=2)
    print("\nTrends for short DataFrame:")
    print(short_df_trends)

    # Test with all NaNs in a window for a value
    nan_df_data = {
        'patient_id': [0,0,0,0],
        'hour': [0,1,2,3],
        'value': [10, np.nan, np.nan, 15]
    }
    nan_df = pd.DataFrame(nan_df_data)
    nan_df_trends = make_trends(nan_df.copy(), 'patient_id', 'hour', ['value'], windows=[3], slope_window_min_periods=2)
    print("\nTrends for DataFrame with NaNs in window:")
    print(nan_df_trends)

    nan_df_all_nan_window_data = {
        'patient_id': [0,0,0,0],
        'hour': [0,1,2,3],
        'value': [np.nan, np.nan, np.nan, 15] # First window of 3 for slope will be all NaN
    }
    nan_df_all_nan = pd.DataFrame(nan_df_all_nan_window_data)
    nan_df_all_nan_trends = make_trends(nan_df_all_nan.copy(), 'patient_id', 'hour', ['value'], windows=[3], slope_window_min_periods=2)
    print("\nTrends for DataFrame with all NaNs in a rolling window for slope:")
    print(nan_df_all_nan_trends)

```
