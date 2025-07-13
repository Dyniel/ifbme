import pandas as pd
import os
import logging
import time # Added for timing

logger = logging.getLogger(__name__)

def load_raw_data(config, base_data_path="data/"):
    """
    Loads raw data (train, val, test) as specified in the config.
    For NCV, it's expected that train and val might be combined externally
    or this function could be called to load them separately before combination.

    Args:
        config (dict): Configuration dictionary containing data paths.
                       Expected structure: config['data_paths']['train'],
                                         config['data_paths']['val'],
                                         config['data_paths']['test']
        base_data_path (str): Base directory where data files are located.

    Returns:
        tuple: Depending on what's loaded, can return (X_train, y_train), (X_val, y_val), etc.
               For simplicity in this skeleton, let's assume it loads a "full" dataset for NCV.
               A more robust version would load specific sets (train, val, test).
               This conceptual version will try to load train and val and combine them.
    """
    logger.info("Attempting to load raw data...")
    overall_start_time = time.time()

    train_file = config.get('data_paths', {}).get('train', 'trainData.csv')
    val_file = config.get('data_paths', {}).get('val', 'valData.csv')

    train_path = os.path.join(base_data_path, train_file)
    val_path = os.path.join(base_data_path, val_file)

    t0 = time.time()
    try:
        df_train = pd.read_csv(train_path)
        logger.info(f"Loaded training data from {train_path}, shape: {df_train.shape}. Time: {time.time() - t0:.2f}s")
    except FileNotFoundError:
        logger.error(f"Training data file not found at {train_path}. Please check config and data directory.")
        raise
    except Exception as e:
        logger.error(f"Error loading training data from {train_path}: {e}")
        raise

    t0 = time.time()
    try:
        df_val = pd.read_csv(val_path)
        logger.info(f"Loaded validation data from {val_path}, shape: {df_val.shape}. Time: {time.time() - t0:.2f}s")
    except FileNotFoundError:
        logger.warning(f"Validation data file not found at {val_path}. Proceeding with training data only for 'full_data' if applicable.")
        df_val = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading validation data from {val_path}: {e}")
        df_val = pd.DataFrame()

    t0 = time.time()
    target_column = config.get('data_paths', {}).get('target_column', 'target')

    if not df_val.empty:
        df_full = pd.concat([df_train, df_val], ignore_index=True)
        logger.info(f"Combined training and validation data. Full data shape: {df_full.shape}. Time: {time.time() - t0:.2f}s")
    else:
        df_full = df_train
        logger.info(f"Using only training data as full data (validation data not loaded/found). Time: {time.time() - t0:.2f}s")

    logger.info(f"Available columns in the loaded DataFrame (df_full): {df_full.columns.tolist()}")
    # ---- END DEBUG PRINT ----

    if target_column not in df_full.columns:
        logger.error(f"Target column '{target_column}' (specified in config) not found in the loaded data.")
        logger.error(f"Please ensure the 'target_column' in your config matches one of the available columns listed above.")
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns: {df_full.columns.tolist()}")

    X_full = df_full.drop(columns=[target_column])
    y_full = df_full[target_column]

    # --- Date Column Conversion ---
    # Identify potential date columns and convert them to numerical (e.g., Unix timestamp)
    date_columns_to_convert = ['requestDate', 'admissionDate']
    date_conversion_start_time = time.time()
    logger.info("Starting date column conversions...")
    for col_name in date_columns_to_convert:
        if col_name in X_full.columns:
            t_col_start = time.time()
            try:
                logger.info(f"Attempting conversion of column '{col_name}'...")
                # Try to infer format for potentially faster parsing if consistent
                # Otherwise, pandas will try to infer, which can be slow.
                # Example: X_full[col_name] = pd.to_datetime(X_full[col_name], format='%d/%m/%Y', errors='coerce')
                # For this dataset, dates seem to be in multiple formats (e.g. YYYY-MM-DD and DD/MM/YYYY)
                # So, direct format specification might lead to more NaTs if not careful.
                # Letting pandas infer, but logging time.
                if col_name == 'requestDate':
                    X_full[col_name] = pd.to_datetime(X_full[col_name], dayfirst=True, errors='coerce')
                else:
                    X_full[col_name] = pd.to_datetime(X_full[col_name], errors='coerce')
                X_full[col_name] = (X_full[col_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                logger.info(f"Converted column '{col_name}' to Unix timestamp. Time: {time.time() - t_col_start:.2f}s")
            except Exception as e:
                logger.warning(f"Could not convert date column '{col_name}' to numeric: {e}. Time: {time.time() - t_col_start:.2f}s. It might remain as object type.")
        else:
            logger.warning(f"Date column '{col_name}' specified for conversion not found in X_full.")
    logger.info(f"Finished date column conversions. Total time: {time.time() - date_conversion_start_time:.2f}s")
    # --- End Date Column Conversion ---

    logger.info(f"Data loading and initial date conversion complete. X_full shape: {X_full.shape}, y_full shape: {y_full.shape}. Total time: {time.time() - overall_start_time:.2f}s")
    return X_full, y_full # Return DataFrames

# Example usage (conceptual, would be in train.py)
# if __name__ == '__main__':
#     dummy_config = {
#         'data_paths': {
#             'train': 'trainData.csv',
#             'val': 'valData.csv',
#             'test': 'testData.csv',
#             'target_column': 'Mortality' # Example target column name
#         }
#     }
#     # Create dummy CSVs for testing
#     if not os.path.exists("data"): os.makedirs("data")
#     pd.DataFrame({'feature1': range(10), 'feature2': range(10,20), 'Mortality': [0]*5 + [1]*5}).to_csv("data/trainData.csv", index=False)
#     pd.DataFrame({'feature1': range(5), 'feature2': range(5,10), 'Mortality': [0]*2 + [1]*3}).to_csv("data/valData.csv", index=False)

#     try:
#         X_full_loaded, y_full_loaded = load_raw_data(dummy_config)
#         logger.info(f"Successfully loaded and combined data: X_full shape {X_full_loaded.shape}, y_full shape {y_full_loaded.shape}")
#     except Exception as e:
#         logger.error(f"Failed to load data in example: {e}")
