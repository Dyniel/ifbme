import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml
import wandb
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

# Project-specific imports
from data_utils.balancing import RSMOTEGAN
from data_utils.data_loader import load_raw_data
from data_utils.preprocess import get_preprocessor
from models import LightGBMModel, XGBoostMetaLearner
from models.teco_transformer import TECOTransformerModel
from data_utils.sequence_loader import TabularSequenceDataset, basic_collate_fn  # For TECO

# --- GNN Imports ---
from data_utils.graph_schema import NODE_TYPES as GNN_NODE_TYPES, EDGE_TYPES as GNN_EDGE_TYPES # Graph Schema
from data_utils.graph_loader import PatientHeteroGraphDataset, create_global_mappers
from models.hetero_temporal_gnn import HeteroTemporalGNN
from torch_geometric.loader import DataLoader as PyGDataLoader # DataLoader for PyG graph batches
# --- End GNN Imports ---


# For STM-GNN, data loading might be more complex (graph snapshots)
# from data_utils.graph_loader import GraphSnapshotDataset, graph_collate_fn # Conceptual - now replaced by PatientHeteroGraphDataset

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise


def main(config_path):
    config = load_config(config_path)

    wandb_config = config.get('wandb', {})
    wandb.init(
        project=wandb_config.get('project', 'ifbme-project'),
        entity=wandb_config.get('entity', None),
        config=config,
        name=wandb_config.get('run_name', "full_pipeline_run"),
        notes=wandb_config.get('run_notes', "Full training pipeline run with NCV."),
        tags=wandb_config.get('tags', ['full_run', 'ncv']),
        mode="offline"  # Ensure W&B runs in offline mode to avoid API key prompts
    )
    logger.info(f"W&B initialized for project '{wandb_config.get('project', 'ifbme-project')}' in offline mode.")
    print("WANDB init completed.")  # For debugging
    sys.stdout.flush()  # Ensure it prints immediately

    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu")
    logger.info(f"Using device: {device}")

    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    logger.info("Starting data loading and preprocessing for NCV...")
    use_dummy_data = config.get('use_dummy_data_for_full_run', False)  # Control real vs dummy data

    if use_dummy_data:
        logger.info("Using dummy data generation for this run as per 'use_dummy_data_for_full_run' config.")
        num_total_samples = config.get('dummy_data_total_samples', 500)
        num_features = config.get('dummy_data_features', 20)
        num_classes_config = config.get('dummy_data_classes', 2)
        if num_classes_config < 2: raise ValueError("dummy_data_classes must be >= 2.")
        weights_combined = config.get('dummy_data_weights', [0.9, 0.1])
        p_combined = np.array(weights_combined) / np.sum(weights_combined)
        X_full_raw_df = pd.DataFrame(np.random.rand(num_total_samples, num_features),
                                     columns=[f'feature_{i}' for i in range(num_features)])
        y_full_raw_series = pd.Series(np.random.choice(num_classes_config, num_total_samples, p=p_combined))
        logger.info(
            f"Dummy raw data generated: X_df shape {X_full_raw_df.shape}, y_series shape {y_full_raw_series.shape}")
    else:
        try:
            logger.info("Loading real data...")
            X_full_raw_df, y_full_raw_series = load_raw_data(config, base_data_path=config.get('data_dir', 'data/'))
            logger.info(f"Real data loaded: X_df shape {X_full_raw_df.shape}, y_series shape {y_full_raw_series.shape}")
        except Exception as e:
            logger.critical(f"Failed to load real data: {e}. Exiting.")
            sys.exit(1)

    unique_classes_full_raw = np.unique(y_full_raw_series)
    num_classes = len(unique_classes_full_raw)
    logger.info(
        f"Determined number of classes for NCV: {num_classes} from y_full_raw unique values: {unique_classes_full_raw}")
    if num_classes < 2:
        raise ValueError(f"Number of classes must be at least 2. Found {num_classes}.")

    # --- GNN Global Setup (Mappers & Patient ID Column) ---
    gnn_config = config.get('ensemble', {}).get('gnn_params', {})
    train_gnn = config.get('ensemble', {}).get('train_gnn', False)
    global_concept_mappers = None
    patient_id_col_name_for_gnn = config.get('patient_id_column') # Get from top-level config

    if train_gnn:
        # Generate 'graph_instance_id' if specified in config, or use existing patient_id_column
        # This allows flexibility: either use a pre-existing unique ID per row/encounter,
        # or generate one if each row is an independent graph unit.
        # The YAML should set patient_id_column to 'graph_instance_id' if generation is desired.
        if patient_id_col_name_for_gnn == 'graph_instance_id': # Special value to trigger generation
            logger.info(f"Generating '{patient_id_col_name_for_gnn}' for GNN processing.")
            X_full_raw_df = X_full_raw_df.reset_index(drop=True)
            X_full_raw_df[patient_id_col_name_for_gnn] = X_full_raw_df.index
            y_full_raw_series.index = X_full_raw_df.index # Align y_series index
            logger.info(f"'{patient_id_col_name_for_gnn}' column created and y_series index aligned.")
        elif patient_id_col_name_for_gnn not in X_full_raw_df.columns and X_full_raw_df.index.name != patient_id_col_name_for_gnn:
            logger.error(f"Specified GNN patient ID column '{patient_id_col_name_for_gnn}' not found in X_full_raw_df columns or as index name. Disabling GNN.")
            train_gnn = False
        elif X_full_raw_df.index.name == patient_id_col_name_for_gnn:
             # If patient_id_col_name is the index, make it a regular column for consistent handling downstream
             # (e.g. in PatientHeteroGraphDataset's y_map creation, create_global_mappers)
             X_full_raw_df[patient_id_col_name_for_gnn] = X_full_raw_df.index
             logger.info(f"Using index '{patient_id_col_name_for_gnn}' as GNN patient ID column and made it a regular column.")


        if train_gnn: # Re-check after potential modification of X_full_raw_df
            logger.info("GNN training is enabled. Creating global concept mappers...")
            gnn_data_cols = gnn_config.get('data_columns', {})

            vital_cols_for_gnn = gnn_data_cols.get('vital_columns', [])
            diag_col_for_gnn = gnn_data_cols.get('diagnosis_column')
            med_col_for_gnn = gnn_data_cols.get('medication_column') # Might be None if not configured
            proc_col_for_gnn = gnn_data_cols.get('procedure_column') # Might be None if not configured
            timestamp_col_for_gnn = gnn_data_cols.get('event_timestamp_column')

            # Check essential GNN column configurations
            # Medication and Procedure columns are optional for mapper creation
            if not all([patient_id_col_name_for_gnn, vital_cols_for_gnn, diag_col_for_gnn, timestamp_col_for_gnn]):
                logger.error("Missing critical GNN data column configurations (patient_id, vitals, diagnosis, timestamp) in YAML. Disabling GNN.")
                train_gnn = False
            else:
                try:
                    global_concept_mappers = create_global_mappers(
                        all_patient_data_df=X_full_raw_df.copy(), # Pass a copy
                        patient_id_col=patient_id_col_name_for_gnn,
                        vital_col_names=vital_cols_for_gnn,
                        diagnosis_col_name=diag_col_for_gnn,
                        medication_col_name=med_col_for_gnn, # Pass it, create_global_mappers will handle if None/missing
                        procedure_col_name=proc_col_for_gnn, # Pass it
                        timestamp_col=timestamp_col_for_gnn
                    )
                    logger.info("Global concept mappers created for GNN.")
                except Exception as e_map:
                    logger.error(f"Error creating global concept mappers for GNN: {e_map}. Disabling GNN training.")
                    logger.error(traceback.format_exc())
                    train_gnn = False

    if not train_gnn:
        logger.info("GNN training is disabled for this run.")

import time # Added for timing (should be at the top, but ensuring it's present)

# ... (other imports)

# --- Preprocessing Setup (for tabular models like LGBM, TECO) ---
    logger.info("Starting preprocessing setup...")
    preproc_setup_start_time = time.time()

    preproc_cfg = config.get('preprocessing', {})
    numerical_cols_from_config = preproc_cfg.get('numerical_cols', [])
    categorical_cols_from_config = preproc_cfg.get('categorical_cols', [])

    if isinstance(X_full_raw_df, pd.DataFrame):
        all_df_columns_set = set(X_full_raw_df.columns.tolist())

        # Initialize from config or auto-detect if config lists are empty/not provided
        if numerical_cols_from_config or categorical_cols_from_config:
            numerical_cols = [col for col in numerical_cols_from_config if col in all_df_columns_set]
            categorical_cols = [col for col in categorical_cols_from_config if col in all_df_columns_set]

            if not numerical_cols_from_config and categorical_cols_from_config:  # Only categorical provided
                numerical_cols = [col for col in all_df_columns_set if col not in categorical_cols]
            elif not categorical_cols_from_config and numerical_cols_from_config:  # Only numerical provided
                categorical_cols = [col for col in all_df_columns_set if col not in numerical_cols]
            # If both are provided (or both empty meaning use all other columns), they are already set.
            # If both config lists are empty, it implies user wants auto-detection for unspecified columns,
            # but if they provided empty lists, it might mean "no columns of this type".
            # This part of logic might need refinement if config can specify empty list to mean "no such columns".
            # Current assumption: empty list in config = auto-detect for that type if other list is also empty/not given.

        else:  # Neither numerical_cols nor categorical_cols provided in config, so auto-detect all.
            logger.info("No numerical/categorical column lists in config. Auto-detecting all based on dtype.")
            numerical_cols = X_full_raw_df.select_dtypes(include=np.number).columns.tolist()
            # Include 'category' dtype as categorical by default as well
            categorical_cols = X_full_raw_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Ensure initial disjointness: if a column is in both (e.g. from config error or complex auto-detection), prefer categorical.
        common_cols = list(set(numerical_cols) & set(categorical_cols))
        if common_cols:
            logger.warning(
                f"Columns {common_cols} initially identified in both numerical and categorical. Defaulting them to categorical for safety.")
            numerical_cols = [col for col in numerical_cols if col not in common_cols]

    else:  # Not a DataFrame, rely purely on config lists
        numerical_cols = list(numerical_cols_from_config)
        categorical_cols = list(categorical_cols_from_config)
        common_cols_non_df = list(set(numerical_cols) & set(categorical_cols))
        if common_cols_non_df:
            logger.warning(
                f"Columns {common_cols_non_df} found in both numerical and categorical lists (non-DataFrame input). Defaulting to categorical.")
            numerical_cols = [col for col in numerical_cols if col not in common_cols_non_df]

    logger.info(f"Initial numerical_cols (config/auto-detected): {numerical_cols}")
    logger.info(f"Initial categorical_cols (config/auto-detected): {categorical_cols}")

    # Robust handling of column types and cleaning for DataFrames
    if isinstance(X_full_raw_df, pd.DataFrame):
        logger.info("Cleaning numerical columns and re-assigning types if necessary...")

        known_placeholders = ['Not applicable', 'NA', 'N/A', '', ' ', 'NaN', 'nan',
                              '<NA>']  # Common string representations of NaN

        cols_to_move_to_categorical = []
        final_cleaned_numerical_cols = []

        for col_name in list(numerical_cols):  # Iterate over a copy of the current numerical_cols list
            if col_name not in X_full_raw_df.columns:
                logger.warning(f"Numerical column '{col_name}' not found in DataFrame. Skipping.")
                continue

            col_series = X_full_raw_df[col_name].copy()  # Work on a copy of the series

            # If the column is of object type, it might contain string placeholders for NaNs or actual string values.
            if col_series.dtype == 'object' or pd.api.types.is_categorical_dtype(col_series.dtype):
                for placeholder in known_placeholders:
                    col_series.replace(placeholder, np.nan, inplace=True)

            # Attempt to convert to a numeric type. errors='coerce' turns unparseable values into NaN.
            converted_series = pd.to_numeric(col_series, errors='coerce')

            # Check the result of the conversion
            if pd.api.types.is_numeric_dtype(converted_series):
                # If original was object/category AND all values became NaN, it implies the column was full of non-numeric strings.
                # Such columns are better treated as categorical.
                if (X_full_raw_df[col_name].dtype == 'object' or pd.api.types.is_categorical_dtype(
                        X_full_raw_df[col_name])) and converted_series.isnull().all():
                    logger.warning(
                        f"Column '{col_name}' (original type: {X_full_raw_df[col_name].dtype}) was designated numerical but became all NaNs after cleaning. Moving to categorical.")
                    cols_to_move_to_categorical.append(col_name)
                    X_full_raw_df[col_name] = converted_series  # Keep the NaNs for categorical imputer
                else:
                    # Successfully converted (or was already numeric), update the DataFrame.
                    X_full_raw_df[col_name] = converted_series
                    final_cleaned_numerical_cols.append(col_name)
            else:
                # Failed to convert to a numeric type even with errors='coerce' (should be rare, implies complex objects).
                # Or, pd.to_numeric might return object type if it contains mixed non-numeric data not handled by coerce (e.g. datetimes, timedeltas if not handled earlier)
                logger.warning(
                    f"Column '{col_name}' (original type: {X_full_raw_df[col_name].dtype}) could not be converted to a pure numeric type (resulting dtype: {converted_series.dtype}). Moving to categorical.")
                cols_to_move_to_categorical.append(col_name)
                # Ensure the column in the DataFrame reflects NaNs for unparseable parts if it was object
                if X_full_raw_df[col_name].dtype == 'object':
                    X_full_raw_df[col_name] = converted_series  # converted_series here would have NaNs for unparseable

        numerical_cols = final_cleaned_numerical_cols

        # Add columns that were moved to the categorical list
        for col_to_move in cols_to_move_to_categorical:
            if col_to_move not in categorical_cols:
                categorical_cols.append(col_to_move)

        # Clean placeholders in all columns now considered categorical
        for col_name_cat in list(categorical_cols):
            if col_name_cat in X_full_raw_df.columns:
                if X_full_raw_df[col_name_cat].dtype == 'object' or pd.api.types.is_categorical_dtype(
                        X_full_raw_df[col_name_cat]):
                    col_cat_series = X_full_raw_df[col_name_cat].copy()
                    for placeholder in known_placeholders:
                        col_cat_series.replace(placeholder, np.nan, inplace=True)
                    X_full_raw_df[col_name_cat] = col_cat_series

        # Final pass to ensure lists are unique and disjoint, and columns exist
        all_df_columns_set = set(X_full_raw_df.columns.tolist())
        numerical_cols = sorted(list(set(col for col in numerical_cols if col in all_df_columns_set)))
        categorical_cols = sorted(list(set(col for col in categorical_cols if col in all_df_columns_set)))

        final_common_cols = list(set(numerical_cols) & set(categorical_cols))
        if final_common_cols:
            logger.warning(
                f"Columns {final_common_cols} ended up in both lists after cleaning. Prioritizing as categorical.")
            numerical_cols = [col for col in numerical_cols if col not in final_common_cols]

        logger.info(f"Final numerical columns after cleaning: {numerical_cols}")
        logger.info(f"Final categorical columns after cleaning (includes reassigned): {categorical_cols}")

    # Global preprocessor - will be fit on outer train folds
    # Note: For NCV, the preprocessor should be fit *inside each outer fold* on its training split.
    # This `global_preprocessor` is a template; a new one is instantiated per outer fold.

    # --- Nested Cross-Validation Setup ---
    n_outer_folds = config.get('ensemble', {}).get('n_outer_folds', 5)
    n_inner_folds = config.get('ensemble', {}).get('n_inner_folds_for_oof', 5)
    outer_skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)

    outer_fold_metrics_meta = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': []}
    outer_fold_metrics_soft_vote = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': []}

    X_full_for_split = X_full_raw_df

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_full_raw_series_encoded = pd.Series(le.fit_transform(y_full_raw_series), name=y_full_raw_series.name,
                                          index=y_full_raw_series.index)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(
        f"Target variable '{y_full_raw_series.name}' encoded. Mapping: {class_mapping}. Unique values after encoding: {y_full_raw_series_encoded.unique()}")
    y_full_for_split = y_full_raw_series_encoded

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
            outer_skf.split(X_full_raw_df, y_full_for_split)):
        logger.info(f"===== Starting Outer Fold {outer_fold_idx + 1}/{n_outer_folds} =====")

        X_outer_train_raw_fold_df = X_full_raw_df.iloc[outer_train_idx]
        y_outer_train_fold_series = y_full_for_split.iloc[outer_train_idx]
        X_outer_test_raw_fold_df = X_full_raw_df.iloc[outer_test_idx]
        y_outer_test_fold_series = y_full_for_split.iloc[outer_test_idx]

        logger.debug(
            f"Outer Fold {outer_fold_idx + 1}: X_outer_train_raw_fold_df shape {X_outer_train_raw_fold_df.shape}, X_outer_test_raw_fold_df shape {X_outer_test_raw_fold_df.shape}")

        fold_preprocessor = get_preprocessor(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            imputation_strategy=preproc_cfg.get('imputation_strategy', 'median'),
            scale_numerics=preproc_cfg.get('scale_numerics', True),
            handle_unknown_categorical=preproc_cfg.get('onehot_handle_unknown', 'ignore')
        )
        try:
            logger.info(f"Outer Fold {outer_fold_idx + 1}: Fitting preprocessor on X_outer_train_raw_fold_df...")
            X_outer_train_processed = fold_preprocessor.fit_transform(X_outer_train_raw_fold_df)
            X_outer_test_processed = fold_preprocessor.transform(X_outer_test_raw_fold_df)

            try:
                processed_feature_names = fold_preprocessor.get_feature_names_out()
            except Exception:
                num_processed_features = X_outer_train_processed.shape[1]
                processed_feature_names = [f'proc_feat_{i}' for i in range(num_processed_features)]
                logger.warning(
                    f"Could not get feature names from preprocessor. Using generic names: {processed_feature_names[:5]}...")

            y_outer_train = y_outer_train_fold_series.to_numpy()
            y_outer_test = y_outer_test_fold_series.to_numpy()
            logger.info(
                f"Outer Fold {outer_fold_idx + 1}: Preprocessing complete. X_outer_train_processed shape {X_outer_train_processed.shape}, X_outer_test_processed shape {X_outer_test_processed.shape}")
        except Exception as e:
            logger.error(
                f"Outer Fold {outer_fold_idx + 1}: Error during general preprocessing: {e}. Using raw data for this outer fold (if numpy).")
            X_outer_train_processed = X_outer_train_raw_fold_df.to_numpy() if isinstance(X_outer_train_raw_fold_df,
                                                                                         pd.DataFrame) else X_outer_train_raw_fold_df
            X_outer_test_processed = X_outer_test_raw_fold_df.to_numpy() if isinstance(X_outer_test_raw_fold_df,
                                                                                       pd.DataFrame) else X_outer_test_raw_fold_df
            y_outer_train = y_outer_train_fold_series.to_numpy()
            y_outer_test = y_outer_test_fold_series.to_numpy()
            processed_feature_names = X_outer_train_raw_fold_df.columns.tolist() if isinstance(
                X_outer_train_raw_fold_df, pd.DataFrame) else [f'raw_feat_{i}' for i in
                                                               range(X_outer_train_processed.shape[1])]

        inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + outer_fold_idx)
        oof_preds_inner = {
            'lgbm': np.zeros((len(y_outer_train), num_classes)),
            'teco': np.zeros((len(y_outer_train), num_classes)),
            # Add GNN entry if training GNN
        }
        base_model_preds_on_outer_test_sum = {
            'lgbm': np.zeros((len(y_outer_test), num_classes)),
            'teco': np.zeros((len(y_outer_test), num_classes)),
            # Add GNN entry if training GNN
        }
        if train_gnn:
            oof_preds_inner['gnn'] = np.zeros((len(y_outer_train), num_classes)) # Assuming num_classes for GNN output
            base_model_preds_on_outer_test_sum['gnn'] = np.zeros((len(y_outer_test), num_classes))


        # --- GNN Dataset Instantiation for Outer Test Fold (if GNN is active) ---
        # This dataset is used by the GNN model trained on the full outer_train split
        # to make predictions on the outer_test data.
        outer_test_graph_dataset = None
        if train_gnn and global_concept_mappers is not None:
            try:
                logger.info(f"Outer Fold {outer_fold_idx + 1}: Preparing GNN Dataset for X_outer_test_raw_fold_df...")
                # Ensure y_outer_test_fold_series is indexed by patient_id for y_map construction
                # This part is tricky: y_map needs patient_id -> (label, label_timestamp_abs)
                # We need to reconstruct this from X_outer_test_raw_fold_df and y_outer_test_fold_series
                # Assume patient_id_col_name is the index of X_outer_test_raw_fold_df for simplicity here
                # And label_timestamp_col is present in X_outer_test_raw_fold_df (e.g. 'dischargeDate' or similar)

                # This y_map construction needs to be robust.
                # It assumes X_outer_test_raw_fold_df is indexed by patient_id_col_name
                # and y_outer_test_fold_series is also indexed by patient_id_col_name.
                # It also needs a reliable 'label_timestamp_col' (e.g. 'dischargeDate') in X_outer_test_raw_fold_df.

                # Placeholder for y_map for test set (label can be dummy, timestamp is important)
                y_map_outer_test = {}
                if X_outer_test_raw_fold_df.index.name != patient_id_col_name:
                     X_outer_test_raw_fold_df_indexed = X_outer_test_raw_fold_df.set_index(patient_id_col_name, drop=False)
                else:
                     X_outer_test_raw_fold_df_indexed = X_outer_test_raw_fold_df

                for pid_test in X_outer_test_raw_fold_df_indexed.index.unique():
                    # This assumes label_timestamp_col is available in X_outer_test_raw_fold_df_indexed
                    # and represents the event time for which a prediction is made (e.g., discharge time)
                    # Using the first available timestamp for that patient as a proxy for label event time. This is a simplification.
                    label_ts_val = X_outer_test_raw_fold_df_indexed.loc[pid_test, gnn_config['data_columns']['label_timestamp_column']].iloc[0] \
                        if isinstance(X_outer_test_raw_fold_df_indexed.loc[pid_test, gnn_config['data_columns']['label_timestamp_column']], pd.Series) \
                        else X_outer_test_raw_fold_df_indexed.loc[pid_test, gnn_config['data_columns']['label_timestamp_column']]

                    y_map_outer_test[pid_test] = (y_outer_test_fold_series.loc[pid_test] if pid_test in y_outer_test_fold_series.index else 0,
                                                  pd.to_datetime(label_ts_val))

                gnn_construction_params_outer = gnn_config.get('graph_construction_params', {}).copy()
                gnn_construction_params_outer['global_concept_mappers'] = global_concept_mappers

                outer_test_graph_dataset = PatientHeteroGraphDataset(
                    root_dir=os.path.join(config.get('output_dir', 'outputs'), f'fold_{outer_fold_idx+1}', 'gnn_processed_test'),
                    patient_df_split=X_outer_test_raw_fold_df, # Raw features for this outer test fold
                    patient_id_col=patient_id_col_name,
                    y_map=y_map_outer_test, # Map of patient_id to (label, label_timestamp_abs)
                    target_variable_name=y_full_for_split.name, # Original target column name
                    label_timestamp_col=gnn_config['data_columns']['label_timestamp_column'], # To identify event time for snapshot
                    timestamp_col=gnn_config['data_columns']['event_timestamp_column'],
                    time_rel_col_name=gnn_config['data_columns'].get('relative_time_column', 'hours_since_admission'),
                    admission_timestamp_col=gnn_config['data_columns']['admission_timestamp_column'],
                    graph_construction_params=gnn_construction_params_outer,
                    vital_col_names=gnn_config['data_columns']['vital_columns'],
                    diagnosis_col_name=gnn_config['data_columns']['diagnosis_column'],
                    medication_col_name=gnn_config['data_columns']['medication_column'],
                    procedure_col_name=gnn_config['data_columns']['procedure_column'],
                    force_reprocess=gnn_config.get('force_reprocess_graphs', False)
                )
            except Exception as e_ds_test:
                logger.error(f"Outer Fold {outer_fold_idx + 1}: Error creating GNN test dataset: {e_ds_test}. GNN predictions for outer test will be defaults.")
                outer_test_graph_dataset = None # Ensure it's None if creation fails
        # --- End GNN Dataset for Outer Test ---

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
                inner_skf.split(X_outer_train_processed, y_outer_train)):
            logger.info(
                f"--- Starting Inner Fold {inner_fold_idx + 1}/{n_inner_folds} (Outer Fold {outer_fold_idx + 1}) ---")
            X_inner_fold_train, y_inner_fold_train = X_outer_train_processed[inner_train_idx], y_outer_train[
                inner_train_idx]
            X_inner_fold_val, y_inner_fold_val = X_outer_train_processed[inner_val_idx], y_outer_train[inner_val_idx]

            if config.get('balancing', {}).get('use_rsmote_gan_in_cv', True):
                logger.info(f"Inner Fold {inner_fold_idx + 1}: Applying RSMOTE-GAN...")
                rsmote_cv_config = config['balancing'].get('rsmote_gan_params', {})
                rsmote_gan_cv = RSMOTEGAN(
                    k_neighbors=rsmote_cv_config.get('k', 5),
                    minority_upsample_factor=rsmote_cv_config.get('minority_upsample_factor', 3.0),
                    random_state=seed + outer_fold_idx + inner_fold_idx
                )
                try:
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = rsmote_gan_cv.fit_resample(
                        X_inner_fold_train, y_inner_fold_train)
                    logger.info(
                        f"Inner Fold {inner_fold_idx + 1}: RSMOTE-GAN completed. New shape: {X_inner_fold_train_balanced.shape}")
                except Exception as e:
                    logger.error(
                        f"Inner Fold {inner_fold_idx + 1}: Error during RSMOTE-GAN: {e}. Proceeding without balancing.")
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = X_inner_fold_train, y_inner_fold_train
            else:
                X_inner_fold_train_balanced, y_inner_fold_train_balanced = X_inner_fold_train, y_inner_fold_train

            if config.get('ensemble', {}).get('train_lgbm', True):
                logger.info(f"Inner Fold {inner_fold_idx + 1}: Training LightGBM...")
                try:
                    lgbm_config = config.get('ensemble', {}).get('lgbm_params', {})
                    lgbm_inner_fold_model = LightGBMModel(
                        params=lgbm_config.get('model_specific_params'),
                        num_leaves=lgbm_config.get('num_leaves', 10000),
                        class_weight=lgbm_config.get('class_weight', 'balanced')
                    )
                    lgbm_inner_fold_model.train(
                        X_inner_fold_train_balanced, y_inner_fold_train_balanced,
                        X_inner_fold_val, y_inner_fold_val,
                        num_boost_round=lgbm_config.get('num_boost_round', 1000),
                        early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 50)
                    )

                    # Get probabilities
                    lgbm_oof_probas_raw = lgbm_inner_fold_model.predict_proba(X_inner_fold_val)
                    lgbm_test_probas_raw = lgbm_inner_fold_model.predict_proba(X_outer_test_processed)

                    # Ensure 2D shape for binary classification [prob_class_0, prob_class_1]
                    if num_classes == 2:
                        if lgbm_oof_probas_raw.ndim == 1:
                            lgbm_oof_probas_2d = np.vstack([1 - lgbm_oof_probas_raw, lgbm_oof_probas_raw]).T
                        elif lgbm_oof_probas_raw.shape[1] == 1:  # Handles cases where it might return (N,1)
                            lgbm_oof_probas_2d = np.hstack([1 - lgbm_oof_probas_raw, lgbm_oof_probas_raw])
                        else:
                            lgbm_oof_probas_2d = lgbm_oof_probas_raw

                        if lgbm_test_probas_raw.ndim == 1:
                            lgbm_test_probas_2d = np.vstack([1 - lgbm_test_probas_raw, lgbm_test_probas_raw]).T
                        elif lgbm_test_probas_raw.shape[1] == 1:
                            lgbm_test_probas_2d = np.hstack([1 - lgbm_test_probas_raw, lgbm_test_probas_raw])
                        else:
                            lgbm_test_probas_2d = lgbm_test_probas_raw
                    else:  # Multiclass, should already be (N, num_classes)
                        lgbm_oof_probas_2d = lgbm_oof_probas_raw
                        lgbm_test_probas_2d = lgbm_test_probas_raw

                    # Assign to OOF predictions array
                    if oof_preds_inner['lgbm'][inner_val_idx].shape == lgbm_oof_probas_2d.shape:
                        oof_preds_inner['lgbm'][inner_val_idx] = lgbm_oof_probas_2d
                    else:
                        logger.error(
                            f"LGBM OOF shape mismatch: Target shape {oof_preds_inner['lgbm'][inner_val_idx].shape}, Value shape {lgbm_oof_probas_2d.shape}. Filling with default.")
                        oof_preds_inner['lgbm'][inner_val_idx] = np.full((len(inner_val_idx), num_classes),
                                                                         1 / num_classes)

                    # Add to base model predictions for outer test set
                    if base_model_preds_on_outer_test_sum['lgbm'].shape == lgbm_test_probas_2d.shape:
                        base_model_preds_on_outer_test_sum['lgbm'] += lgbm_test_probas_2d / n_inner_folds
                    else:
                        logger.error(
                            f"LGBM Test Sum shape mismatch: Target shape {base_model_preds_on_outer_test_sum['lgbm'].shape}, Value shape {lgbm_test_probas_2d.shape}. Adding default.")
                        base_model_preds_on_outer_test_sum['lgbm'] += np.full((len(y_outer_test), num_classes),
                                                                              1 / num_classes) / n_inner_folds

                    logger.info(f"Inner Fold {inner_fold_idx + 1}: LightGBM training and prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during LightGBM: {e}")
                    oof_preds_inner['lgbm'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1 / num_classes)
                    base_model_preds_on_outer_test_sum['lgbm'] += np.full((len(y_outer_test), num_classes),
                                                                          1 / num_classes) / n_inner_folds

            if config.get('ensemble', {}).get('train_teco', True):
                logger.info(f"Inner Fold {inner_fold_idx + 1}: Training TECO-Transformer...")
                try:
                    teco_config = config.get('ensemble', {}).get('teco_params', {})

                    # --- TECO Debug Logging Start ---
                    logger.debug(
                        f"TECO: X_inner_fold_train_balanced type: {type(X_inner_fold_train_balanced)}, shape: {X_inner_fold_train_balanced.shape if isinstance(X_inner_fold_train_balanced, np.ndarray) else 'N/A'}")
                    logger.debug(
                        f"TECO: y_inner_fold_train_balanced type: {type(y_inner_fold_train_balanced)}, shape: {y_inner_fold_train_balanced.shape if isinstance(y_inner_fold_train_balanced, np.ndarray) else 'N/A'}")
                    if isinstance(X_inner_fold_train_balanced, np.ndarray) and X_inner_fold_train_balanced.size > 0:
                        logger.debug(
                            f"TECO: X_inner_fold_train_balanced sample (first 5): \n{X_inner_fold_train_balanced[:5, :min(5, X_inner_fold_train_balanced.shape[1])]} \ndtypes: {X_inner_fold_train_balanced.dtype}")
                    if isinstance(y_inner_fold_train_balanced, np.ndarray) and y_inner_fold_train_balanced.size > 0:
                        logger.debug(
                            f"TECO: y_inner_fold_train_balanced sample (first 5): {y_inner_fold_train_balanced[:5]}, dtype: {y_inner_fold_train_balanced.dtype}")

                    logger.debug(
                        f"TECO: X_inner_fold_val type: {type(X_inner_fold_val)}, shape: {X_inner_fold_val.shape if isinstance(X_inner_fold_val, np.ndarray) else 'N/A'}")
                    logger.debug(
                        f"TECO: y_inner_fold_val type: {type(y_inner_fold_val)}, shape: {y_inner_fold_val.shape if isinstance(y_inner_fold_val, np.ndarray) else 'N/A'}")

                    logger.debug(
                        f"TECO: processed_feature_names type: {type(processed_feature_names)}, len: {len(processed_feature_names) if isinstance(processed_feature_names, list) else 'N/A'}")
                    if isinstance(processed_feature_names, list) and len(processed_feature_names) > 0:
                        logger.debug(f"TECO: processed_feature_names sample (first 5): {processed_feature_names[:5]}")
                    # --- End TECO Debug Logging ---

                    df_inner_fold_train_teco = pd.DataFrame(X_inner_fold_train_balanced,
                                                            columns=processed_feature_names)
                    df_inner_fold_val_teco = pd.DataFrame(X_inner_fold_val, columns=processed_feature_names)
                    df_outer_test_teco = pd.DataFrame(X_outer_test_processed, columns=processed_feature_names)

                    teco_target_column_name = 'target_for_teco'

                    # --- TECO Debug Logging for DataFrames ---
                    logger.debug(
                        f"TECO: df_inner_fold_train_teco shape: {df_inner_fold_train_teco.shape}, dtypes head: \n{df_inner_fold_train_teco.dtypes.head()}")
                    if not df_inner_fold_train_teco.empty:
                        logger.debug(
                            f"TECO: df_inner_fold_train_teco sample head: \n{df_inner_fold_train_teco.head(2)}")
                    # --- End TECO Debug Logging ---

                    train_teco_dataset_inner = TabularSequenceDataset(
                        data_frame=df_inner_fold_train_teco,
                        targets=y_inner_fold_train_balanced,
                        feature_columns=processed_feature_names,
                        target_column_name=teco_target_column_name
                    )
                    val_teco_dataset_inner = TabularSequenceDataset(df_inner_fold_val_teco, y_inner_fold_val,
                                                                    processed_feature_names, teco_target_column_name)
                    outer_test_teco_dataset = TabularSequenceDataset(df_outer_test_teco,
                                                                     np.zeros(len(df_outer_test_teco)),
                                                                     # Dummy targets for test
                                                                     processed_feature_names, teco_target_column_name)

                    batch_size_teco = teco_config.get('batch_size_teco', 32)
                    train_teco_loader_inner = DataLoader(train_teco_dataset_inner, batch_size=batch_size_teco,
                                                         shuffle=True, collate_fn=basic_collate_fn)
                    val_teco_loader_inner = DataLoader(val_teco_dataset_inner, batch_size=batch_size_teco,
                                                       shuffle=False, collate_fn=basic_collate_fn)
                    outer_test_teco_loader = DataLoader(outer_test_teco_dataset, batch_size=batch_size_teco,
                                                        shuffle=False, collate_fn=basic_collate_fn)

                    teco_model_inner = TECOTransformerModel(
                        input_feature_dim=len(processed_feature_names),
                        d_model=teco_config.get('d_model', 512),
                        num_encoder_layers=teco_config.get('num_encoder_layers', 4),
                        nhead=teco_config.get('nhead', 8),
                        dim_feedforward=teco_config.get('dim_feedforward', 2048),
                        dropout=teco_config.get('dropout', 0.1),
                        num_classes=num_classes,
                        max_seq_len=teco_config.get('max_seq_len', 100)
                    ).to(device)

                    teco_criterion_inner = nn.CrossEntropyLoss()
                    teco_optimizer_inner = optim.Adam(teco_model_inner.parameters(),
                                                      lr=teco_config.get('lr_teco', 1e-4))
                    epochs_teco_inner = teco_config.get('epochs_teco_inner', 10)

                    # --- TECO Training Phase ---
                    try:
                        for epoch in range(epochs_teco_inner):
                            teco_model_inner.train()
                            epoch_loss_sum = 0.0
                            for batch_idx, batch in enumerate(train_teco_loader_inner):
                                teco_optimizer_inner.zero_grad()
                                sequences = batch['sequence'].to(device)
                                padding_masks = batch['padding_mask'].to(device)
                                targets = batch['target'].to(device)

                                outputs = teco_model_inner(sequences, padding_masks)
                                loss = teco_criterion_inner(outputs, targets)
                                loss.backward()
                                teco_optimizer_inner.step()
                                epoch_loss_sum += loss.item()
                            logger.debug(
                                f"Inner Fold {inner_fold_idx + 1}, TECO Epoch {epoch + 1}/{epochs_teco_inner}, Avg Train Loss: {epoch_loss_sum / len(train_teco_loader_inner):.4f}")
                    except Exception as e_train_teco:
                        logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during TECO Training Loop: {e_train_teco}")
                        logger.error(f"Error occurred in TECO training, epoch {epoch+1 if 'epoch' in locals() else 'unknown'}, batch_idx {batch_idx if 'batch_idx' in locals() else 'unknown'}")
                        raise  # Re-raise to be caught by the outer TECO try-except

                    # --- TECO Validation Prediction Phase ---
                    try:
                        teco_model_inner.eval()
                        inner_val_preds_teco_list = []
                        with torch.no_grad():
                            for batch in val_teco_loader_inner:
                                outputs = teco_model_inner(batch['sequence'].to(device), batch['padding_mask'].to(device))
                                inner_val_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

                        if inner_val_preds_teco_list:
                            oof_preds_inner['teco'][inner_val_idx] = np.concatenate(inner_val_preds_teco_list, axis=0)[:, :num_classes]
                        else:
                            logger.warning(f"Inner Fold {inner_fold_idx + 1}: TECO validation prediction list is empty. Filling with defaults.")
                            oof_preds_inner['teco'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1 / num_classes)
                    except Exception as e_val_pred_teco:
                        logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during TECO Validation Prediction: {e_val_pred_teco}")
                        raise # Re-raise to be caught by the outer TECO try-except

                    # --- TECO Outer Test Prediction Phase ---
                    try:
                        outer_test_preds_teco_list = []
                        with torch.no_grad():
                            for batch in outer_test_teco_loader:
                                outputs = teco_model_inner(batch['sequence'].to(device), batch['padding_mask'].to(device))
                                outer_test_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

                        if outer_test_preds_teco_list:
                            base_model_preds_on_outer_test_sum['teco'] += np.concatenate(outer_test_preds_teco_list, axis=0)[:, :num_classes] / n_inner_folds
                        else:
                            logger.warning(f"Inner Fold {inner_fold_idx + 1}: TECO outer test prediction list is empty. Adding defaults.")
                            base_model_preds_on_outer_test_sum['teco'] += np.full((len(y_outer_test), num_classes), 1 / num_classes) / n_inner_folds
                    except Exception as e_test_pred_teco:
                        logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during TECO Outer Test Prediction: {e_test_pred_teco}")
                        raise # Re-raise to be caught by the outer TECO try-except

                    logger.info(f"Inner Fold {inner_fold_idx + 1}: TECO-Transformer training and prediction complete.")
                except Exception as e: # This is the main TECO exception handler (line 548 in original)
                    logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during TECO-Transformer (Main Block): {e}")
                    # Ensure traceback is logged for unexpected errors not caught by more specific blocks above
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    oof_preds_inner['teco'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1 / num_classes)
                    base_model_preds_on_outer_test_sum['teco'] += np.full((len(y_outer_test), num_classes),
                                                                          1 / num_classes) / n_inner_folds

            # --- GNN Training and Prediction within Inner Fold ---
            if train_gnn and global_concept_mappers is not None:
                logger.info(f"Inner Fold {inner_fold_idx + 1}: Training HeteroTemporalGNN...")
                try:
                    # Prepare GNN datasets for this inner fold
                    # X_outer_train_raw_fold_df needs to be split into inner_train_raw and inner_val_raw
                    # These DFs are used to instantiate PatientHeteroGraphDataset

                    # Important: X_outer_train_raw_fold_df contains the *original features*, not the preprocessed ones.
                    # We need to use inner_train_idx and inner_val_idx on X_outer_train_raw_fold_df and y_outer_train_fold_series

                    X_inner_train_raw_gnn = X_outer_train_raw_fold_df.iloc[inner_train_idx]
                    y_inner_train_gnn_series = y_outer_train_fold_series.iloc[inner_train_idx]

                    X_inner_val_raw_gnn = X_outer_train_raw_fold_df.iloc[inner_val_idx]
                    y_inner_val_gnn_series = y_outer_train_fold_series.iloc[inner_val_idx]

                    # Construct y_map for inner train and val GNN datasets
                    y_map_inner_train_gnn = {pid: (y_inner_train_gnn_series.loc[pid], pd.to_datetime(X_inner_train_raw_gnn.loc[pid, gnn_config['data_columns']['label_timestamp_column']]))
                                             for pid in X_inner_train_raw_gnn.index.unique() if pid in y_inner_train_gnn_series.index}
                    y_map_inner_val_gnn = {pid: (y_inner_val_gnn_series.loc[pid], pd.to_datetime(X_inner_val_raw_gnn.loc[pid, gnn_config['data_columns']['label_timestamp_column']]))
                                           for pid in X_inner_val_raw_gnn.index.unique() if pid in y_inner_val_gnn_series.index}

                    gnn_construction_params_inner = gnn_config.get('graph_construction_params', {}).copy()
                    gnn_construction_params_inner['global_concept_mappers'] = global_concept_mappers

                    inner_train_graph_dataset = PatientHeteroGraphDataset(
                        root_dir=os.path.join(config.get('output_dir', 'outputs'), f'fold_{outer_fold_idx+1}_inner_{inner_fold_idx+1}', 'gnn_processed_train'),
                        patient_df_split=X_inner_train_raw_gnn,
                        patient_id_col=patient_id_col_name, y_map=y_map_inner_train_gnn,
                        target_variable_name=y_full_for_split.name,
                        label_timestamp_col=gnn_config['data_columns']['label_timestamp_column'],
                        timestamp_col=gnn_config['data_columns']['event_timestamp_column'],
                        time_rel_col_name=gnn_config['data_columns'].get('relative_time_column', 'hours_since_admission'),
                        admission_timestamp_col=gnn_config['data_columns']['admission_timestamp_column'],
                        graph_construction_params=gnn_construction_params_inner,
                        vital_col_names=gnn_config['data_columns']['vital_columns'],
                        diagnosis_col_name=gnn_config['data_columns']['diagnosis_column'],
                        medication_col_name=gnn_config['data_columns']['medication_column'],
                        procedure_col_name=gnn_config['data_columns']['procedure_column'],
                        force_reprocess=gnn_config.get('force_reprocess_graphs', False)
                    )
                    inner_val_graph_dataset = PatientHeteroGraphDataset(
                        root_dir=os.path.join(config.get('output_dir', 'outputs'), f'fold_{outer_fold_idx+1}_inner_{inner_fold_idx+1}', 'gnn_processed_val'),
                        patient_df_split=X_inner_val_raw_gnn,
                        patient_id_col=patient_id_col_name, y_map=y_map_inner_val_gnn,
                        # ... other params same as inner_train_graph_dataset ...
                        target_variable_name=y_full_for_split.name,
                        label_timestamp_col=gnn_config['data_columns']['label_timestamp_column'],
                        timestamp_col=gnn_config['data_columns']['event_timestamp_column'],
                        time_rel_col_name=gnn_config['data_columns'].get('relative_time_column', 'hours_since_admission'),
                        admission_timestamp_col=gnn_config['data_columns']['admission_timestamp_column'],
                        graph_construction_params=gnn_construction_params_inner,
                        vital_col_names=gnn_config['data_columns']['vital_columns'],
                        diagnosis_col_name=gnn_config['data_columns']['diagnosis_column'],
                        medication_col_name=gnn_config['data_columns']['medication_column'],
                        procedure_col_name=gnn_config['data_columns']['procedure_column'],
                        force_reprocess=gnn_config.get('force_reprocess_graphs', False)
                    )

                    if not inner_train_graph_dataset.patient_ids or not inner_val_graph_dataset.patient_ids:
                        raise ValueError("GNN dataset for inner fold resulted in no patients.")

                    gnn_batch_size = gnn_config.get('batch_size', 32)
                    # PyG DataLoader for graph data
                    train_gnn_loader = PyGDataLoader(inner_train_graph_dataset, batch_size=gnn_batch_size, shuffle=True)
                    val_gnn_loader = PyGDataLoader(inner_val_graph_dataset, batch_size=gnn_batch_size, shuffle=False)

                    # GNN Model Instantiation
                    # Need to get timeslice_feat_dim from a sample graph or config
                    # This is data['timeslice'].x.shape[1]
                    # For now, placeholder - this needs to be robustly determined.
                    # Example: sample_graph_data = inner_train_graph_dataset[0]
                    # timeslice_input_dim_gnn = sample_graph_data['timeslice'].x.shape[1]

                    # This is a temporary hack to get the feature dimension.
                    # A more robust way would be to define it based on time_embedding_dim + num_vital_features
                    # from graph_constructor and make it available.
                    # For now, let's assume it's configured or can be inferred if datasets are non-empty.
                    timeslice_input_dim_gnn = gnn_config.get('timeslice_feature_dim', 16 + len(gnn_config['data_columns']['vital_columns'])) # Placeholder calculation
                    if inner_train_graph_dataset.patient_ids:
                        try:
                            sample_graph_data_for_dim = inner_train_graph_dataset.get(0) # Use .get() to load from disk if processed
                            if sample_graph_data_for_dim and 'timeslice' in sample_graph_data_for_dim.node_types and sample_graph_data_for_dim['timeslice'].num_nodes > 0:
                                timeslice_input_dim_gnn = sample_graph_data_for_dim['timeslice'].x.shape[1]
                            else:
                                logger.warning("Sample graph for GNN timeslice dim is empty/invalid. Using configured/default.")
                        except Exception as e_sample_dim:
                             logger.warning(f"Could not get sample graph for GNN timeslice dim: {e_sample_dim}. Using configured/default.")


                    gnn_model_inner = HeteroTemporalGNN(
                        data_schema={'NODE_TYPES': GNN_NODE_TYPES, 'EDGE_TYPES': GNN_EDGE_TYPES}, # Pass the schema
                        num_nodes_dict={ntype: len(mapper) for ntype, mapper in global_concept_mappers.items()},
                        timeslice_feat_dim=timeslice_input_dim_gnn,
                        concept_embedding_dim=gnn_config.get('concept_embedding_dim', 64),
                        gnn_hidden_dim=gnn_config.get('gnn_hidden_dim', 128),
                        gnn_output_dim=gnn_config.get('gnn_output_dim', 128), # Output of GNN layers before final FC
                        num_gnn_layers=gnn_config.get('num_gnn_layers', 2),
                        num_gat_heads=gnn_config.get('num_gat_heads', 4),
                        output_classes=1 if num_classes == 2 else num_classes, # BCEWithLogitsLoss if binary
                        dropout_rate=gnn_config.get('dropout', 0.3)
                    ).to(device)

                    gnn_criterion_inner = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
                    gnn_optimizer_inner = optim.Adam(gnn_model_inner.parameters(), lr=gnn_config.get('lr', 1e-3))
                    epochs_gnn_inner = gnn_config.get('epochs_inner', 10)

                    # --- GNN Training Loop ---
                    for epoch in range(epochs_gnn_inner):
                        gnn_model_inner.train()
                        epoch_loss_gnn_sum = 0.0
                        for gnn_batch in train_gnn_loader:
                            gnn_batch = gnn_batch.to(device)
                            gnn_optimizer_inner.zero_grad()
                            gnn_outputs = gnn_model_inner(gnn_batch) # HeteroData batch
                            gnn_loss = gnn_criterion_inner(gnn_outputs, gnn_batch.y.float()) # Ensure y is float for BCE
                            gnn_loss.backward()
                            gnn_optimizer_inner.step()
                            epoch_loss_gnn_sum += gnn_loss.item()
                        logger.debug(f"Inner Fold {inner_fold_idx + 1}, GNN Epoch {epoch + 1}/{epochs_gnn_inner}, Avg Train Loss: {epoch_loss_gnn_sum / len(train_gnn_loader):.4f}")

                    # --- GNN Validation Predictions (OOF for meta-learner) ---
                    gnn_model_inner.eval()
                    inner_val_preds_gnn_list = []
                    with torch.no_grad():
                        for gnn_batch_val in val_gnn_loader:
                            gnn_batch_val = gnn_batch_val.to(device)
                            gnn_outputs_val = gnn_model_inner(gnn_batch_val)
                            # Sigmoid for BCEWithLogitsLoss, then get class 1 prob
                            gnn_probs_val = torch.sigmoid(gnn_outputs_val).cpu().numpy()
                            inner_val_preds_gnn_list.append(gnn_probs_val)

                    if inner_val_preds_gnn_list:
                        concatenated_preds_gnn = np.concatenate(inner_val_preds_gnn_list)
                        # Ensure 2D for binary: [prob_class_0, prob_class_1]
                        if num_classes == 2:
                             gnn_oof_probas_2d = np.hstack([1 - concatenated_preds_gnn.reshape(-1,1), concatenated_preds_gnn.reshape(-1,1)])
                        else: # Multiclass: ensure shape is (N, num_classes) - GNN output needs adjustment if not
                             gnn_oof_probas_2d = concatenated_preds_gnn # This assumes GNN outputs (N,num_classes) for multiclass

                        if oof_preds_inner['gnn'][inner_val_idx].shape == gnn_oof_probas_2d.shape:
                             oof_preds_inner['gnn'][inner_val_idx] = gnn_oof_probas_2d
                        else:
                             logger.error(f"GNN OOF shape mismatch. Target: {oof_preds_inner['gnn'][inner_val_idx].shape}, Value: {gnn_oof_probas_2d.shape}")
                             # Fill with default if mismatch
                             oof_preds_inner['gnn'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1/num_classes)

                    # --- GNN Outer Test Predictions (from this inner fold's GNN model) ---
                    # This requires outer_test_graph_dataset to be ready
                    if outer_test_graph_dataset and len(outer_test_graph_dataset) > 0:
                        outer_test_gnn_loader = PyGDataLoader(outer_test_graph_dataset, batch_size=gnn_batch_size, shuffle=False)
                        outer_test_preds_gnn_list = []
                        with torch.no_grad():
                            for gnn_batch_test in outer_test_gnn_loader:
                                gnn_batch_test = gnn_batch_test.to(device)
                                gnn_outputs_test = gnn_model_inner(gnn_batch_test)
                                gnn_probs_test = torch.sigmoid(gnn_outputs_test).cpu().numpy()
                                outer_test_preds_gnn_list.append(gnn_probs_test)

                        if outer_test_preds_gnn_list:
                            concatenated_test_preds_gnn = np.concatenate(outer_test_preds_gnn_list)
                            if num_classes == 2:
                                gnn_test_probas_2d = np.hstack([1 - concatenated_test_preds_gnn.reshape(-1,1), concatenated_test_preds_gnn.reshape(-1,1)])
                            else:
                                gnn_test_probas_2d = concatenated_test_preds_gnn # Adjust if multiclass output from GNN is different

                            if base_model_preds_on_outer_test_sum['gnn'].shape == gnn_test_probas_2d.shape:
                                base_model_preds_on_outer_test_sum['gnn'] += gnn_test_probas_2d / n_inner_folds
                            else:
                                logger.error(f"GNN Test Sum shape mismatch. Target: {base_model_preds_on_outer_test_sum['gnn'].shape}, Value: {gnn_test_probas_2d.shape}")

                    logger.info(f"Inner Fold {inner_fold_idx + 1}: HeteroTemporalGNN training and prediction complete.")

                except Exception as e_gnn_inner:
                    logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during HeteroTemporalGNN: {e_gnn_inner}")
                    import traceback
                    logger.error(f"GNN Traceback: {traceback.format_exc()}")
                    # Fill with defaults if GNN fails for this inner fold
                    oof_preds_inner['gnn'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1 / num_classes)
                    base_model_preds_on_outer_test_sum['gnn'] += np.full((len(y_outer_test), num_classes), 1 / num_classes) / n_inner_folds
            # --- End GNN Block ---

        logger.info(f"Outer Fold {outer_fold_idx + 1}: Finished generating OOF predictions from inner CV.")
        meta_features_train_outer_list = []
        if config.get('ensemble', {}).get('train_lgbm', True): meta_features_train_outer_list.append(
            oof_preds_inner['lgbm'])
        if config.get('ensemble', {}).get('train_teco', True): meta_features_train_outer_list.append(
            oof_preds_inner['teco'])
        if train_gnn: # Add GNN OOF preds if GNN was trained
            meta_features_train_outer_list.append(oof_preds_inner['gnn'])


        if not meta_features_train_outer_list:
            logger.error(
                f"Outer Fold {outer_fold_idx + 1}: No base models for meta-learner. Skipping meta-learner and soft voting for this fold.")
            for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)
            for key in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key].append(np.nan)
            # Continue to next outer fold
            # wandb.log({f"outer_fold_{outer_fold_idx + 1}/meta_auroc": np.nan, "outer_fold": outer_fold_idx + 1})
            # wandb.log({f"outer_fold_{outer_fold_idx + 1}/sv_auroc": np.nan, "outer_fold": outer_fold_idx + 1})
            continue

        X_meta_train_outer = np.concatenate(meta_features_train_outer_list, axis=1)
        y_meta_train_outer = y_outer_train
        logger.info(
            f"Outer Fold {outer_fold_idx + 1}: Meta-learner training features shape: {X_meta_train_outer.shape}")

        if config.get('ensemble', {}).get('train_meta_learner', True):
            logger.info(f"Outer Fold {outer_fold_idx + 1}: Training XGBoost Meta-Learner...")
            try:
                meta_config = config.get('ensemble', {}).get('meta_learner_xgb_params', {})
                xgb_meta_model_outer = XGBoostMetaLearner(
                    params=meta_config.get('model_specific_params'),
                    depth=meta_config.get('depth', 3)
                )
                xgb_meta_model_outer.train(
                    X_meta_train_outer, y_meta_train_outer,
                    num_boost_round=meta_config.get('num_boost_round', 200),
                    early_stopping_rounds=meta_config.get('early_stopping_rounds', 20)
                )
                logger.info(f"Outer Fold {outer_fold_idx + 1}: XGBoost Meta-Learner trained.")

                meta_features_test_outer_list = []
                if config.get('ensemble', {}).get('train_lgbm', True): meta_features_test_outer_list.append(
                    base_model_preds_on_outer_test_sum['lgbm'])
                if config.get('ensemble', {}).get('train_teco', True): meta_features_test_outer_list.append(
                    base_model_preds_on_outer_test_sum['teco'])
                if train_gnn: # Add GNN test preds if GNN was trained
                    meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['gnn'])

                if not meta_features_test_outer_list: # Should not happen if train_meta_learner is true and at least one base model ran
                    logger.error(f"Outer Fold {outer_fold_idx + 1}: No base model predictions available for meta-learner test set. Skipping meta-learner prediction.")
                    # Fill meta metrics with NaN for this fold
                    for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)
                    wandb.log({f"outer_fold_{outer_fold_idx + 1}/meta_auroc": np.nan, "outer_fold": outer_fold_idx + 1})
                    # Skip to soft voting or next part of the loop for this outer fold
                    # This requires careful restructuring of the following soft-voting block or a continue
                else:
                    X_meta_test_outer = np.concatenate(meta_features_test_outer_list, axis=1)
                    final_preds_meta_proba_outer = xgb_meta_model_outer.predict_proba(X_meta_test_outer)
                final_preds_meta_labels_outer = xgb_meta_model_outer.predict(X_meta_test_outer)

                acc_meta_outer = accuracy_score(y_outer_test, final_preds_meta_labels_outer)
                f1_meta_outer = f1_score(y_outer_test, final_preds_meta_labels_outer,
                                         average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                prec_meta_outer = precision_score(y_outer_test, final_preds_meta_labels_outer,
                                                  average='weighted' if num_classes > 2 else 'binary',
                                                  zero_division=0)
                rec_meta_outer = recall_score(y_outer_test, final_preds_meta_labels_outer,
                                              average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                auroc_meta_outer = -1.0
                try:
                    probas_for_auc = final_preds_meta_proba_outer[:,
                                     1] if num_classes == 2 and final_preds_meta_proba_outer.ndim == 2 and \
                                           final_preds_meta_proba_outer.shape[
                                               1] >= 2 else final_preds_meta_proba_outer
                    auroc_meta_outer = roc_auc_score(y_outer_test, probas_for_auc, multi_class='ovr',
                                                     average='weighted')
                except ValueError as e:
                    logger.warning(
                        f"Outer Fold {outer_fold_idx + 1} Meta AUROC calc error: {e}. Proba shape: {final_preds_meta_proba_outer.shape if isinstance(final_preds_meta_proba_outer, np.ndarray) else 'N/A'}")

                outer_fold_metrics_meta['accuracy'].append(acc_meta_outer)
                outer_fold_metrics_meta['auroc'].append(auroc_meta_outer)
                outer_fold_metrics_meta['f1'].append(f1_meta_outer)
                outer_fold_metrics_meta['precision'].append(prec_meta_outer)
                outer_fold_metrics_meta['recall'].append(rec_meta_outer)
                wandb.log({f"outer_fold_{outer_fold_idx + 1}/meta_auroc": auroc_meta_outer,
                           f"outer_fold_{outer_fold_idx + 1}/meta_acc": acc_meta_outer,
                           "outer_fold": outer_fold_idx + 1})
                logger.info(
                    f"Outer Fold {outer_fold_idx + 1} Meta-Learner: AUROC={auroc_meta_outer:.4f}, Acc={acc_meta_outer:.4f}")
            except Exception as e:
                logger.error(f"Outer Fold {outer_fold_idx + 1}: Error during Meta-Learner: {e}")
                for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)
                wandb.log({f"outer_fold_{outer_fold_idx + 1}/meta_auroc": np.nan, "outer_fold": outer_fold_idx + 1})

        soft_vote_weights = config.get('ensemble', {}).get('soft_vote_weights', {})

        # Determine active models for soft voting, now potentially including GNN
        potential_sv_models = ['lgbm', 'teco']
        if train_gnn:
            potential_sv_models.append('gnn')

        active_models_for_sv = [model_key for model_key in potential_sv_models
                                if config.get('ensemble', {}).get(f'train_{model_key}', True) and \
                                   model_key in soft_vote_weights and \
                                   model_key in base_model_preds_on_outer_test_sum # Ensure predictions exist
                               ]

        if soft_vote_weights and active_models_for_sv:
            logger.info(
                f"Outer Fold {outer_fold_idx + 1}: Performing Soft Voting with models: {active_models_for_sv}...")
            try:
                final_preds_soft_vote_proba_outer = np.zeros((len(y_outer_test), num_classes))
                total_weight = 0.0

                for model_key in active_models_for_sv: # Iterate only over active, weighted models
                    weight = soft_vote_weights.get(model_key, 0) # Should be >0 due to check above
                    # Ensure the predictions are valid before adding
                    preds_to_add = base_model_preds_on_outer_test_sum.get(model_key)

                    if preds_to_add is not None and preds_to_add.shape == final_preds_soft_vote_proba_outer.shape:
                        final_preds_soft_vote_proba_outer += weight * preds_to_add
                        total_weight += weight
                    else:
                        logger.warning(
                            f"Soft Voting: Skipping model {model_key} due to invalid predictions (shape {preds_to_add.shape if preds_to_add is not None else 'None'} vs target {final_preds_soft_vote_proba_outer.shape} or missing).")

                if total_weight > 1e-6: # Check if any valid weighted predictions were actually added
                    # Normalize probabilities if total_weight is not 1 (or close to it)
                    # This is more robust if weights are relative importance rather than summing to 1.
                    final_preds_soft_vote_proba_outer /= total_weight
                    # Ensure probabilities sum to 1 per sample after weighting and summing
                    row_sums = final_preds_soft_vote_proba_outer.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    final_preds_soft_vote_proba_outer = final_preds_soft_vote_proba_outer / row_sums

                    final_preds_soft_vote_labels_outer = np.argmax(final_preds_soft_vote_proba_outer, axis=1)
                    acc_sv_outer = accuracy_score(y_outer_test, final_preds_soft_vote_labels_outer)
                    f1_sv_outer = f1_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                           average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    prec_sv_outer = precision_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                                    average='weighted' if num_classes > 2 else 'binary',
                                                    zero_division=0)
                    rec_sv_outer = recall_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                                average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    auroc_sv_outer = -1.0
                    try:
                        probas_for_auc_sv = final_preds_soft_vote_proba_outer[:,
                                            1] if num_classes == 2 and final_preds_soft_vote_proba_outer.ndim == 2 and \
                                                  final_preds_soft_vote_proba_outer.shape[
                                                      1] >= 2 else final_preds_soft_vote_proba_outer
                        auroc_sv_outer = roc_auc_score(y_outer_test, probas_for_auc_sv, multi_class='ovr',
                                                       average='weighted')
                    except ValueError as e:
                        logger.warning(
                            f"Outer Fold {outer_fold_idx + 1} SoftVote AUROC calculation error: {e}. Proba shape: {final_preds_soft_vote_proba_outer.shape if isinstance(final_preds_soft_vote_proba_outer, np.ndarray) else 'N/A'}")

                    outer_fold_metrics_soft_vote['accuracy'].append(acc_sv_outer)
                    outer_fold_metrics_soft_vote['auroc'].append(auroc_sv_outer)
                    outer_fold_metrics_soft_vote['f1'].append(f1_sv_outer)
                    outer_fold_metrics_soft_vote['precision'].append(prec_sv_outer)
                    outer_fold_metrics_soft_vote['recall'].append(rec_sv_outer)
                    wandb.log(
                        {f"outer_fold_{outer_fold_idx + 1}/sv_auroc": auroc_sv_outer,
                         f"outer_fold_{outer_fold_idx + 1}/sv_acc": acc_sv_outer,
                         "outer_fold": outer_fold_idx + 1})
                    logger.info(
                        f"Outer Fold {outer_fold_idx + 1} Soft Vote: AUROC={auroc_sv_outer:.4f}, Acc={acc_sv_outer:.4f}")
                else:
                    logger.warning(
                        f"Outer Fold {outer_fold_idx + 1}: Soft Voting not performed (no active models with positive weights or zero total weight).")
                    for key_sv in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key_sv].append(
                        np.nan)
                    wandb.log({f"outer_fold_{outer_fold_idx + 1}/sv_auroc": np.nan, "outer_fold": outer_fold_idx + 1})

            except Exception as e:
                logger.error(f"Outer Fold {outer_fold_idx + 1}: Error during Soft Voting: {e}")
                for key_sv_err in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key_sv_err].append(
                    np.nan)
                wandb.log({f"outer_fold_{outer_fold_idx + 1}/sv_auroc": np.nan, "outer_fold": outer_fold_idx + 1})
        elif not soft_vote_weights:
            logger.info(f"Outer Fold {outer_fold_idx + 1}: Soft voting weights not configured. Skipping soft voting.")
            for key_sv_skip in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key_sv_skip].append(
                np.nan)
        elif not active_models_for_sv:
            logger.info(f"Outer Fold {outer_fold_idx + 1}: No active models for soft voting based on config. Skipping.")
            for key_sv_skip_active in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[
                key_sv_skip_active].append(np.nan)

    logger.info("===== Nested Cross-Validation Summary =====")
    if config.get('ensemble', {}).get('train_meta_learner', True) and any(
            not np.isnan(v) for v in outer_fold_metrics_meta['auroc']):
        avg_meta_auroc = np.nanmean(outer_fold_metrics_meta['auroc'])
        std_meta_auroc = np.nanstd(outer_fold_metrics_meta['auroc'])
        logger.info(f"Meta-Learner Average AUROC: {avg_meta_auroc:.4f} +/- {std_meta_auroc:.4f}")
        wandb.summary["ncv_meta_avg_auroc"] = avg_meta_auroc
        wandb.summary["ncv_meta_std_auroc"] = std_meta_auroc
        for metric_name, values in outer_fold_metrics_meta.items():
            if metric_name != 'auroc':
                avg_val = np.nanmean(values)
                wandb.summary[f"ncv_meta_avg_{metric_name}"] = avg_val
                logger.info(f"Meta-Learner Average {metric_name.capitalize()}: {avg_val:.4f}")
    else:
        logger.info("Meta-Learner metrics not computed or all NaN.")
        wandb.summary["ncv_meta_avg_auroc"] = np.nan

    if soft_vote_weights and any(not np.isnan(v) for v in outer_fold_metrics_soft_vote['auroc']):
        avg_sv_auroc = np.nanmean(outer_fold_metrics_soft_vote['auroc'])
        std_sv_auroc = np.nanstd(outer_fold_metrics_soft_vote['auroc'])
        logger.info(f"Soft Voting Average AUROC: {avg_sv_auroc:.4f} +/- {std_sv_auroc:.4f}")
        wandb.summary["ncv_sv_avg_auroc"] = avg_sv_auroc
        wandb.summary["ncv_sv_std_auroc"] = std_sv_auroc
        for metric_name, values in outer_fold_metrics_soft_vote.items():
            if metric_name != 'auroc':
                avg_val = np.nanmean(values)
                wandb.summary[f"ncv_sv_avg_{metric_name}"] = avg_val
                logger.info(f"Soft Voting Average {metric_name.capitalize()}: {avg_val:.4f}")
    else:
        logger.info("Soft Voting metrics not computed or all NaN.")
        wandb.summary["ncv_sv_avg_auroc"] = np.nan

    wandb.finish()
    logger.info("Full Nested Cross-Validation ensemble training run finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main training script for Clinical Prediction Model with NCV.")
    parser.add_argument('--config', type=str, default='configs/dummy_train_config.yaml',
                        help='Path to the training configuration file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} not found. Please create it or provide a valid path.")
        sys.exit(1)

    main(args.config)
