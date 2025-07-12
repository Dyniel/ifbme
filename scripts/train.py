import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
import argparse
import yaml
import wandb
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, mean_absolute_error
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna  # Import Optuna
import joblib

# Project-specific imports
from data_utils.balancing import RSMOTE
from data_utils.data_loader import load_raw_data
from data_utils.losses import FocalLossLGB
from data_utils.preprocess import get_preprocessor
from models import LightGBMModel, XGBoostMetaLearner
from models.teco_transformer import TECOTransformerModel
from data_utils.sequence_loader import TabularSequenceDataset, basic_collate_fn  # For TECO
from utils.metrics import dt_score_calc, ls_score_calc, gl_score_calc, \
    maximise_f1_threshold  # Score calculation functions

# --- GNN Imports ---
from data_utils.graph_schema import NODE_TYPES as GNN_NODE_TYPES, EDGE_TYPES as GNN_EDGE_TYPES  # Graph Schema
from data_utils.graph_loader import PatientHeteroGraphDataset, create_global_mappers
from models.hetero_temporal_gnn import HeteroTemporalGNN
from torch_geometric.loader import DataLoader as PyGDataLoader  # DataLoader for PyG graph batches

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

        # Add dummy text column if text embeddings are enabled in config
        text_embed_cfg_dummy = config.get('text_embedding_params', {})
        if text_embed_cfg_dummy.get('enabled', False):
            dummy_text_col_name = text_embed_cfg_dummy.get('text_column', 'text_notes')
            dummy_texts = [
                "Patient presents with fever and cough.",
                "History of hypertension and diabetes.",
                "No acute distress, vitals stable.",
                "Chest x-ray shows bilateral infiltrates, consider pneumonia.",
                "Plan for discharge tomorrow morning."
            ]
            X_full_raw_df[dummy_text_col_name] = np.random.choice(dummy_texts, num_total_samples)
            logger.info(f"Added dummy text column '{dummy_text_col_name}' to dummy data.")

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
    patient_id_col_name_for_gnn = config.get('patient_id_column')  # Get from top-level config

    if train_gnn:
        # Generate 'graph_instance_id' if specified in config, or use existing patient_id_column
        # This allows flexibility: either use a pre-existing unique ID per row/encounter,
        # or generate one if each row is an independent graph unit.
        # The YAML should set patient_id_column to 'graph_instance_id' if generation is desired.
        if patient_id_col_name_for_gnn == 'graph_instance_id':  # Special value to trigger generation
            logger.info(f"Generating '{patient_id_col_name_for_gnn}' for GNN processing.")
            X_full_raw_df = X_full_raw_df.reset_index(drop=True)
            X_full_raw_df[patient_id_col_name_for_gnn] = X_full_raw_df.index
            y_full_raw_series.index = X_full_raw_df.index  # Align y_series index
            logger.info(f"'{patient_id_col_name_for_gnn}' column created and y_series index aligned.")
        elif patient_id_col_name_for_gnn not in X_full_raw_df.columns and X_full_raw_df.index.name != patient_id_col_name_for_gnn:
            logger.error(
                f"Specified GNN patient ID column '{patient_id_col_name_for_gnn}' not found in X_full_raw_df columns or as index name. Disabling GNN.")
            train_gnn = False
        elif X_full_raw_df.index.name == patient_id_col_name_for_gnn:
            # If patient_id_col_name is the index, make it a regular column for consistent handling downstream
            # (e.g. in PatientHeteroGraphDataset's y_map creation, create_global_mappers)
            X_full_raw_df[patient_id_col_name_for_gnn] = X_full_raw_df.index
            logger.info(
                f"Using index '{patient_id_col_name_for_gnn}' as GNN patient ID column and made it a regular column.")

        if train_gnn:  # Re-check after potential modification of X_full_raw_df
            logger.info("GNN training is enabled. Creating global concept mappers...")
            gnn_data_cols = gnn_config.get('data_columns', {})

            vital_cols_for_gnn = gnn_data_cols.get('vital_columns', [])
            diag_col_for_gnn = gnn_data_cols.get('diagnosis_column')
            med_col_for_gnn = gnn_data_cols.get('medication_column')  # Might be None if not configured
            proc_col_for_gnn = gnn_data_cols.get('procedure_column')  # Might be None if not configured
            timestamp_col_for_gnn = gnn_data_cols.get('event_timestamp_column')

            # Check essential GNN column configurations
            # Medication and Procedure columns are optional for mapper creation
            if not all([patient_id_col_name_for_gnn, vital_cols_for_gnn, diag_col_for_gnn, timestamp_col_for_gnn]):
                logger.error(
                    "Missing critical GNN data column configurations (patient_id, vitals, diagnosis, timestamp) in YAML. Disabling GNN.")
                train_gnn = False
            else:
                try:
                    global_concept_mappers = create_global_mappers(
                        all_patient_data_df=X_full_raw_df.copy(),  # Pass a copy
                        patient_id_col=patient_id_col_name_for_gnn,
                        vital_col_names=vital_cols_for_gnn,
                        diagnosis_col_name=diag_col_for_gnn,
                        medication_col_name=med_col_for_gnn,
                        # Pass it, create_global_mappers will handle if None/missing
                        procedure_col_name=proc_col_for_gnn,  # Pass it
                        timestamp_col=timestamp_col_for_gnn
                    )
                    logger.info("Global concept mappers created for GNN.")
                except Exception as e_map:
                    logger.error(f"Error creating global concept mappers for GNN: {e_map}. Disabling GNN training.")
                    logger.error(traceback.format_exc())
                    train_gnn = False

    if not train_gnn:
        logger.info("GNN training is disabled for this run.")

    # ... (other imports)

    # --- Text Embedding Generation (Optional) ---
    text_embed_config = config.get('text_embedding_params', {})
    if text_embed_config.get('enabled', False):
        logger.info("Text embedding generation is ENABLED.")
        try:
            from features.text_embeddings import get_clinical_bert_model_and_tokenizer, get_text_embeddings
            text_col_name = text_embed_config.get('text_column')
            if text_col_name and text_col_name in X_full_raw_df.columns:
                logger.info(f"Processing text embeddings for column: '{text_col_name}'")

                text_data = X_full_raw_df[text_col_name].fillna("").tolist() # Ensure no NaNs

                model_name = text_embed_config.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT')
                tokenizer, model = get_clinical_bert_model_and_tokenizer(model_name, device=device)

                embeddings = get_text_embeddings(
                    texts=text_data,
                    tokenizer=tokenizer,
                    model=model,
                    strategy=text_embed_config.get('pooling_strategy', 'cls'),
                    max_length=text_embed_config.get('max_length', 512),
                    batch_size=text_embed_config.get('batch_size', 32),
                    device=device
                )

                if embeddings is not None and embeddings.shape[0] == len(X_full_raw_df):
                    embedding_feature_names = [f'text_emb_{i}' for i in range(embeddings.shape[1])]
                    embeddings_df = pd.DataFrame(embeddings, columns=embedding_feature_names, index=X_full_raw_df.index)

                    X_full_raw_df = pd.concat([X_full_raw_df.drop(columns=[text_col_name]), embeddings_df], axis=1)
                    logger.info(f"Successfully added {len(embedding_feature_names)} text embedding features.")

                    # Automatically add new embedding features to numerical_cols in config if not already there
                    if 'preprocessing' not in config: config['preprocessing'] = {}
                    if 'numerical_cols' not in config['preprocessing']: config['preprocessing']['numerical_cols'] = []

                    existing_num_cols = set(config['preprocessing']['numerical_cols'])
                    for emb_col in embedding_feature_names:
                        if emb_col not in existing_num_cols:
                            config['preprocessing']['numerical_cols'].append(emb_col)
                    logger.info("Updated config's numerical_cols with new text embedding features.")

                else:
                    logger.error("Text embedding generation failed or returned incorrect shape. Skipping feature addition.")

            else:
                logger.warning(f"Text embedding column '{text_col_name}' not found in data or not configured. Skipping.")

        except ImportError as e:
            logger.error(f"Failed to import text embedding modules. Make sure transformers are installed. Error: {e}")
        except Exception as e:
            logger.error(f"An error occurred during text embedding generation: {e}")
    else:
        logger.info("Text embedding generation is DISABLED.")

    # --- Ontology Embedding Generation (Optional) ---
    ontology_embed_config = config.get('ontology_embedding_params', {})
    if ontology_embed_config.get('enabled', False):
        logger.info("Ontology embedding generation is ENABLED.")
        try:
            from features.ontology_embeddings import OntologyEmbedder

            code_columns_to_embed = ontology_embed_config.get('code_columns', [])
            if not code_columns_to_embed:
                logger.warning("Ontology embedding enabled, but no 'code_columns' specified in config. Skipping.")
            else:
                embedder = OntologyEmbedder(
                    ontology_data=ontology_embed_config.get('ontology_data'),
                    node2vec_params=ontology_embed_config.get('node2vec_params', {}),
                    pretrained_embeddings_path=ontology_embed_config.get('pretrained_path')
                )

                if ontology_embed_config.get('save_path') and not ontology_embed_config.get('pretrained_path'):
                    embedder.save_embeddings(ontology_embed_config['save_path'])

                for col_name in code_columns_to_embed:
                    if col_name in X_full_raw_df.columns:
                        logger.info(f"Generating ontology embeddings for column: '{col_name}'")
                        codes = X_full_raw_df[col_name].astype(str).tolist()
                        embeddings = embedder.get_embeddings_for_codes(codes)

                        if embeddings is not None and embeddings.shape[0] == len(X_full_raw_df):
                            emb_feature_names = [f'ont_emb_{col_name}_{i}' for i in range(embeddings.shape[1])]
                            embeddings_df = pd.DataFrame(embeddings, columns=emb_feature_names, index=X_full_raw_df.index)

                            X_full_raw_df = pd.concat([X_full_raw_df.drop(columns=[col_name]), embeddings_df], axis=1)
                            logger.info(f"Successfully added {len(emb_feature_names)} ontology embedding features for '{col_name}'.")

                            # Add new features to numerical_cols in config
                            if 'preprocessing' not in config: config['preprocessing'] = {}
                            if 'numerical_cols' not in config['preprocessing']: config['preprocessing']['numerical_cols'] = []

                            existing_num_cols = set(config['preprocessing']['numerical_cols'])
                            for emb_col in emb_feature_names:
                                if emb_col not in existing_num_cols:
                                    config['preprocessing']['numerical_cols'].append(emb_col)
                            logger.info(f"Updated config's numerical_cols for '{col_name}' embeddings.")
                        else:
                            logger.error(f"Ontology embedding generation for '{col_name}' failed. Skipping.")
                    else:
                        logger.warning(f"Ontology code column '{col_name}' not found in data. Skipping.")

        except ImportError as e:
            logger.error(f"Failed to import ontology embedding modules. Make sure node2vec and networkx are installed. Error: {e}")
        except Exception as e:
            logger.error(f"An error occurred during ontology embedding generation: {e}")
    else:
        logger.info("Ontology embedding generation is DISABLED.")


    # --- Preprocessing Setup (for tabular models like LGBM, TECO) ---
    logger.info("Starting preprocessing setup...")  # This line was indented
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

    outer_fold_metrics_meta = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': [], 'dt_score': [],
                               'gl_score': []}
    outer_fold_metrics_soft_vote = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': [],
                                    'dt_score': [], 'gl_score': []}
    outer_fold_los_metrics = {'ls_score': [], 'mae_los': []}  # For Length of Stay specific metrics

    # For accumulating predictions and actuals for final CSV logging
    all_test_indices_list = []
    all_y_test_list = []
    all_preds_meta_list = []  # For DTestimation.csv (predicted outcomeType based on default 0.5 threshold or model's predict())
    all_probas_meta_list = []  # For DTestimation.csv (predicted probabilities for 'Death' class, to allow custom thresholding later)
    all_actual_los_list = []  # For LSestimation.csv (actual lengthOfStay)
    all_predicted_los_list = []  # For LSestimation.csv (predicted lengthOfStay)
    # If patient IDs are consistently available and aligned with X_full_raw_df.index:
    all_patient_ids_list = []
    thr_dict = {}  # For storing best F1 threshold from each fold
    all_f1_at_best_thr_meta_list = []  # For storing F1 score at best threshold for meta-learner from each fold

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
            oof_preds_inner['gnn'] = np.zeros((len(y_outer_train), num_classes))  # Assuming num_classes for GNN output
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
                    X_outer_test_raw_fold_df_indexed = X_outer_test_raw_fold_df.set_index(patient_id_col_name,
                                                                                          drop=False)
                else:
                    X_outer_test_raw_fold_df_indexed = X_outer_test_raw_fold_df

                for pid_test in X_outer_test_raw_fold_df_indexed.index.unique():
                    # This assumes label_timestamp_col is available in X_outer_test_raw_fold_df_indexed
                    # and represents the event time for which a prediction is made (e.g., discharge time)
                    # Using the first available timestamp for that patient as a proxy for label event time. This is a simplification.
                    label_ts_val = X_outer_test_raw_fold_df_indexed.loc[
                        pid_test, gnn_config['data_columns']['label_timestamp_column']].iloc[0] \
                        if isinstance(X_outer_test_raw_fold_df_indexed.loc[
                                          pid_test, gnn_config['data_columns']['label_timestamp_column']], pd.Series) \
                        else X_outer_test_raw_fold_df_indexed.loc[
                        pid_test, gnn_config['data_columns']['label_timestamp_column']]

                    y_map_outer_test[pid_test] = (
                        y_outer_test_fold_series.loc[pid_test] if pid_test in y_outer_test_fold_series.index else 0,
                        pd.to_datetime(label_ts_val))

                gnn_construction_params_outer = gnn_config.get('graph_construction_params', {}).copy()
                gnn_construction_params_outer['global_concept_mappers'] = global_concept_mappers

                outer_test_graph_dataset = PatientHeteroGraphDataset(
                    root_dir=os.path.join(config.get('output_dir', 'outputs'), f'fold_{outer_fold_idx + 1}',
                                          'gnn_processed_test'),
                    patient_df_split=X_outer_test_raw_fold_df,  # Raw features for this outer test fold
                    patient_id_col=patient_id_col_name,
                    y_map=y_map_outer_test,  # Map of patient_id to (label, label_timestamp_abs)
                    target_variable_name=y_full_for_split.name,  # Original target column name
                    label_timestamp_col=gnn_config['data_columns']['label_timestamp_column'],
                    # To identify event time for snapshot
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
                logger.error(
                    f"Outer Fold {outer_fold_idx + 1}: Error creating GNN test dataset: {e_ds_test}. GNN predictions for outer test will be defaults.")
                outer_test_graph_dataset = None  # Ensure it's None if creation fails
        # --- End GNN Dataset for Outer Test ---

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
                inner_skf.split(X_outer_train_processed, y_outer_train)):
            logger.info(
                f"--- Starting Inner Fold {inner_fold_idx + 1}/{n_inner_folds} (Outer Fold {outer_fold_idx + 1}) ---")
            X_inner_fold_train, y_inner_fold_train = X_outer_train_processed[inner_train_idx], y_outer_train[
                inner_train_idx]
            X_inner_fold_val, y_inner_fold_val = X_outer_train_processed[inner_val_idx], y_outer_train[inner_val_idx]

            if config.get('balancing', {}).get('use_rsmote_gan_in_cv', True):
                logger.info(f"Inner Fold {inner_fold_idx + 1}: Applying RSMOTE...")
                rsmote_cv_config = config['balancing'].get('rsmote_gan_params', {})
                rsmote = RSMOTE(
                    k_neighbors=3,
                    random_state=seed + outer_fold_idx + inner_fold_idx
                )
                try:
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = rsmote.fit_resample(
                        X_inner_fold_train, y_inner_fold_train)
                    logger.info(
                        f"Inner Fold {inner_fold_idx + 1}: RSMOTE completed. New shape: {X_inner_fold_train_balanced.shape}")
                    # Log class distribution after balancing
                    unique_classes_balanced, counts_balanced = np.unique(y_inner_fold_train_balanced,
                                                                         return_counts=True)
                    balanced_dist_log_msg = f"Inner Fold {inner_fold_idx + 1} (Outer {outer_fold_idx + 1}) - Class distribution after RSMOTE: {dict(zip(unique_classes_balanced, counts_balanced))}"
                    logger.info(balanced_dist_log_msg)
                    wandb.log({
                        f"outer_fold_{outer_fold_idx + 1}/inner_fold_{inner_fold_idx + 1}/balanced_class_distribution": {
                            str(k): v for k, v in zip(unique_classes_balanced, counts_balanced)},
                        "outer_fold": outer_fold_idx + 1,  # For grouping if needed
                        "inner_fold": inner_fold_idx + 1  # For grouping if needed
                    })
                except Exception as e:
                    logger.error(
                        f"Inner Fold {inner_fold_idx + 1}: Error during RSMOTE: {e}. Proceeding without balancing.")
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = X_inner_fold_train, y_inner_fold_train
            else:
                X_inner_fold_train_balanced, y_inner_fold_train_balanced = X_inner_fold_train, y_inner_fold_train

            if config.get('ensemble', {}).get('train_lgbm', True):
                logger.info(f"Inner Fold {inner_fold_idx + 1}: Training LightGBM with Optuna...")
                try:
                    lgbm_config = config.get('ensemble', {}).get('lgbm_params', {})
                    optuna_lgbm_config = config.get('optuna', {}).get('lgbm', {})
                    n_trials_lgbm = optuna_lgbm_config.get('n_trials', 20)

                    def lgbm_objective(trial):
                        # Define hyperparameter search space for Optuna
                        # Parameters for Focal Loss (CB-Focal)
                        lgbm_params = {
                            'n_estimators': trial.suggest_int('n_estimators',
                                                              lgbm_config.get('num_boost_round', 1000) // 5,
                                                              lgbm_config.get('num_boost_round', 1000) * 2),
                            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                            'num_leaves': trial.suggest_int('num_leaves', 20, lgbm_config.get('num_leaves', 31) * 5,
                                                            log=True),
                            'max_depth': trial.suggest_int('max_depth', -1, 15),
                            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                            # 'class_weight': lgbm_config.get('class_weight', 'balanced'), # Focal loss handles imbalance
                            'random_state': seed + outer_fold_idx + inner_fold_idx,
                            'n_jobs': -1,
                            'verbose': -1,
                        }
                        # Assuming binary classification for 'Death' vs 'Survival' when Focal Loss is applied.
                        # If num_classes > 2 and Focal Loss is still desired, this objective part would need adjustment
                        # or a multiclass Focal Loss version. For now, this setup assumes binary context for Focal.
                        if num_classes > 2:
                            # This part might be problematic if 'binary' objective with alpha/gamma is strictly for 2 classes.
                            # Reverting to default multiclass if Focal params are not compatible.
                            # For now, we assume LightGBMModel or underlying custom objective handles this.
                            # If using standard LightGBM, 'binary' objective with num_class > 2 is an error.
                            # Let's assume for now user ensures LightGBMModel handles this.
                            # However, for safety, if num_classes > 2, Focal Loss might not be applicable directly
                            # with 'binary' objective. This implies the task is primarily binary for Focal Loss.
                            logger.warning(
                                "Focal Loss parameters (alpha, gamma) typically used with 'binary' objective. "
                                f"Current num_classes = {num_classes}. Ensure LightGBMModel handles this if multiclass.")
                            lgbm_params[
                                'num_class'] = num_classes  # Keep if model supports 'binary' objective with num_class for some custom focal setup
                            # If it's truly multiclass and focal is desired, objective might need to be 'multiclass'
                            # and the custom focal loss function should handle multiclass.
                            # For now, sticking to 'binary' as per user note "binary:focal".

                        fobj = FocalLossLGB(alpha=0.25, gamma=2.0)
                        # Train with early stopping on validation set
                        # The train method of LightGBMModel needs to return the score for Optuna
                        # For now, we assume it trains and we get score from predict_proba
                        # This might need adjustment in LightGBMModel or here to directly get best score

                        # We need a temporary model instance for each trial
                        temp_lgbm_model = LightGBMModel(params={'objective': 'binary'})
                        temp_lgbm_model.train(
                            X_inner_fold_train_balanced, y_inner_fold_train_balanced,
                            X_val=X_inner_fold_val, y_val=y_inner_fold_val,
                            fobj=fobj,
                            # num_boost_round passed via n_estimators in lgbm_params
                            early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 20)
                            # Shorter for Optuna trials
                        )

                        preds_proba = temp_lgbm_model.predict_proba(X_inner_fold_val)

                        if num_classes == 2:
                            # Use AUROC for binary classification as it's often preferred for imbalanced data
                            # And it's a common metric for Optuna to optimize
                            try:
                                score = roc_auc_score(y_inner_fold_val,
                                                      preds_proba if preds_proba.ndim == 1 else preds_proba[:, 1])
                                return score  # Optuna maximizes this
                            except ValueError:  # Handle cases where only one class is present in y_inner_fold_val
                                return 0.0  # Return a poor score
                        else:
                            # For multiclass, logloss is fine, Optuna minimizes this
                            # This part needs careful handling of predict_proba output for multi_logloss
                            # For now, let's stick to a placeholder if we need to minimize logloss.
                            # The current LightGBMModel.train uses binary_logloss or multi_logloss for early stopping.
                            # We'll use the best score from the model's internal evaluation if possible, or roc_auc for now.
                            # Placeholder: if multiclass, this needs to be log_loss and Optuna direction set to 'minimize'
                            try:
                                score = roc_auc_score(y_inner_fold_val, preds_proba, multi_class='ovr',
                                                      average='weighted')
                                return score  # Optuna maximizes this
                            except ValueError:
                                return 0.0

                    study_direction = 'maximize' if num_classes == 2 else 'maximize'  # Assuming AUROC for both for now
                    study_lgbm = optuna.create_study(direction=study_direction,
                                                     sampler=optuna.samplers.TPESampler(
                                                         seed=seed + outer_fold_idx + inner_fold_idx))
                    study_lgbm.optimize(lgbm_objective, n_trials=n_trials_lgbm,
                                        timeout=optuna_lgbm_config.get('timeout_seconds_per_fold',
                                                                       600))  # Add a timeout

                    best_params_lgbm = study_lgbm.best_params
                    best_score_lgbm = study_lgbm.best_value
                    logger.info(
                        f"Inner Fold {inner_fold_idx + 1}: Best LGBM params from Optuna: {best_params_lgbm}, Best Score (AUROC/LogLoss): {best_score_lgbm}")
                    wandb.log({
                        f"outer_fold_{outer_fold_idx + 1}/inner_fold_{inner_fold_idx + 1}/lgbm_best_params": best_params_lgbm,
                        f"outer_fold_{outer_fold_idx + 1}/inner_fold_{inner_fold_idx + 1}/lgbm_best_score": best_score_lgbm
                    })

                    # Train final LGBM model for this inner fold using best params
                    # Ensure Focal Loss parameters are included if not already optimized by Optuna (or if Optuna is off)
                    final_lgbm_params = {
                        'random_state': seed + outer_fold_idx + inner_fold_idx,
                        'n_jobs': -1,
                        'verbose': -1,
                    }
                    if num_classes > 2:
                        # Similar warning/consideration as in Optuna objective for multiclass + binary focal
                        logger.warning(
                            f"Final LGBM model with Focal params: num_classes={num_classes} but objective='binary'. Check compatibility.")
                        final_lgbm_params['num_class'] = num_classes

                    # Update with Optuna's best params. Optuna might override alpha/gamma if they were part of its search space.
                    # If alpha/gamma were fixed (as they are currently above), this just adds other tuned params.
                    final_lgbm_params.update(best_params_lgbm)

                    # Re-assert Focal Loss specific params if Optuna didn't tune them or if we want to enforce them
                    # This is crucial if Optuna was optimizing e.g. 'reg_alpha' and we need our specific 'alpha' for Focal Loss.
                    fobj = FocalLossLGB(alpha=0.25, gamma=2.0)

                    lgbm_inner_fold_model = LightGBMModel(params=final_lgbm_params)
                    lgbm_inner_fold_model.train(
                        X_inner_fold_train_balanced, y_inner_fold_train_balanced,
                        X_inner_fold_val, y_inner_fold_val,
                        fobj=fobj,
                        # num_boost_round is now part of best_params_lgbm as n_estimators
                        early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 50)
                        # Use original early stopping for final model
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
                        logger.error(
                            f"Inner Fold {inner_fold_idx + 1}: Error during TECO Training Loop: {e_train_teco}")
                        logger.error(
                            f"Error occurred in TECO training, epoch {epoch + 1 if 'epoch' in locals() else 'unknown'}, batch_idx {batch_idx if 'batch_idx' in locals() else 'unknown'}")
                        raise  # Re-raise to be caught by the outer TECO try-except

                    # --- TECO Validation Prediction Phase ---
                    try:
                        teco_model_inner.eval()
                        inner_val_preds_teco_list = []
                        with torch.no_grad():
                            for batch in val_teco_loader_inner:
                                outputs = teco_model_inner(batch['sequence'].to(device),
                                                           batch['padding_mask'].to(device))
                                inner_val_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

                        if inner_val_preds_teco_list:
                            oof_preds_inner['teco'][inner_val_idx] = np.concatenate(inner_val_preds_teco_list, axis=0)[
                                                                     :, :num_classes]
                        else:
                            logger.warning(
                                f"Inner Fold {inner_fold_idx + 1}: TECO validation prediction list is empty. Filling with defaults.")
                            oof_preds_inner['teco'][inner_val_idx] = np.full((len(inner_val_idx), num_classes),
                                                                             1 / num_classes)
                    except Exception as e_val_pred_teco:
                        logger.error(
                            f"Inner Fold {inner_fold_idx + 1}: Error during TECO Validation Prediction: {e_val_pred_teco}")
                        raise  # Re-raise to be caught by the outer TECO try-except

                    # --- TECO Outer Test Prediction Phase ---
                    try:
                        outer_test_preds_teco_list = []
                        with torch.no_grad():
                            for batch in outer_test_teco_loader:
                                outputs = teco_model_inner(batch['sequence'].to(device),
                                                           batch['padding_mask'].to(device))
                                outer_test_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

                        if outer_test_preds_teco_list:
                            base_model_preds_on_outer_test_sum['teco'] += np.concatenate(outer_test_preds_teco_list,
                                                                                         axis=0)[:,
                                                                          :num_classes] / n_inner_folds
                        else:
                            logger.warning(
                                f"Inner Fold {inner_fold_idx + 1}: TECO outer test prediction list is empty. Adding defaults.")
                            base_model_preds_on_outer_test_sum['teco'] += np.full((len(y_outer_test), num_classes),
                                                                                  1 / num_classes) / n_inner_folds
                    except Exception as e_test_pred_teco:
                        logger.error(
                            f"Inner Fold {inner_fold_idx + 1}: Error during TECO Outer Test Prediction: {e_test_pred_teco}")
                        raise  # Re-raise to be caught by the outer TECO try-except

                    logger.info(f"Inner Fold {inner_fold_idx + 1}: TECO-Transformer training and prediction complete.")
                except Exception as e:  # This is the main TECO exception handler (line 548 in original)
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
                    y_map_inner_train_gnn = {pid: (y_inner_train_gnn_series.loc[pid], pd.to_datetime(
                        X_inner_train_raw_gnn.loc[pid, gnn_config['data_columns']['label_timestamp_column']]))
                                             for pid in X_inner_train_raw_gnn.index.unique() if
                                             pid in y_inner_train_gnn_series.index}
                    y_map_inner_val_gnn = {pid: (y_inner_val_gnn_series.loc[pid], pd.to_datetime(
                        X_inner_val_raw_gnn.loc[pid, gnn_config['data_columns']['label_timestamp_column']]))
                                           for pid in X_inner_val_raw_gnn.index.unique() if
                                           pid in y_inner_val_gnn_series.index}

                    gnn_construction_params_inner = gnn_config.get('graph_construction_params', {}).copy()
                    gnn_construction_params_inner['global_concept_mappers'] = global_concept_mappers

                    inner_train_graph_dataset = PatientHeteroGraphDataset(
                        root_dir=os.path.join(config.get('output_dir', 'outputs'),
                                              f'fold_{outer_fold_idx + 1}_inner_{inner_fold_idx + 1}',
                                              'gnn_processed_train'),
                        patient_df_split=X_inner_train_raw_gnn,
                        patient_id_col=patient_id_col_name, y_map=y_map_inner_train_gnn,
                        target_variable_name=y_full_for_split.name,
                        label_timestamp_col=gnn_config['data_columns']['label_timestamp_column'],
                        timestamp_col=gnn_config['data_columns']['event_timestamp_column'],
                        time_rel_col_name=gnn_config['data_columns'].get('relative_time_column',
                                                                         'hours_since_admission'),
                        admission_timestamp_col=gnn_config['data_columns']['admission_timestamp_column'],
                        graph_construction_params=gnn_construction_params_inner,
                        vital_col_names=gnn_config['data_columns']['vital_columns'],
                        diagnosis_col_name=gnn_config['data_columns']['diagnosis_column'],
                        medication_col_name=gnn_config['data_columns']['medication_column'],
                        procedure_col_name=gnn_config['data_columns']['procedure_column'],
                        force_reprocess=gnn_config.get('force_reprocess_graphs', False)
                    )
                    inner_val_graph_dataset = PatientHeteroGraphDataset(
                        root_dir=os.path.join(config.get('output_dir', 'outputs'),
                                              f'fold_{outer_fold_idx + 1}_inner_{inner_fold_idx + 1}',
                                              'gnn_processed_val'),
                        patient_df_split=X_inner_val_raw_gnn,
                        patient_id_col=patient_id_col_name, y_map=y_map_inner_val_gnn,
                        # ... other params same as inner_train_graph_dataset ...
                        target_variable_name=y_full_for_split.name,
                        label_timestamp_col=gnn_config['data_columns']['label_timestamp_column'],
                        timestamp_col=gnn_config['data_columns']['event_timestamp_column'],
                        time_rel_col_name=gnn_config['data_columns'].get('relative_time_column',
                                                                         'hours_since_admission'),
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
                    timeslice_input_dim_gnn = gnn_config.get('timeslice_feature_dim', 16 + len(
                        gnn_config['data_columns']['vital_columns']))  # Placeholder calculation
                    if inner_train_graph_dataset.patient_ids:
                        try:
                            sample_graph_data_for_dim = inner_train_graph_dataset.get(
                                0)  # Use .get() to load from disk if processed
                            if sample_graph_data_for_dim and 'timeslice' in sample_graph_data_for_dim.node_types and \
                                    sample_graph_data_for_dim['timeslice'].num_nodes > 0:
                                timeslice_input_dim_gnn = sample_graph_data_for_dim['timeslice'].x.shape[1]
                            else:
                                logger.warning(
                                    "Sample graph for GNN timeslice dim is empty/invalid. Using configured/default.")
                        except Exception as e_sample_dim:
                            logger.warning(
                                f"Could not get sample graph for GNN timeslice dim: {e_sample_dim}. Using configured/default.")

                    gnn_model_inner = HeteroTemporalGNN(
                        data_schema={'NODE_TYPES': GNN_NODE_TYPES, 'EDGE_TYPES': GNN_EDGE_TYPES},  # Pass the schema
                        num_nodes_dict={ntype: len(mapper) for ntype, mapper in global_concept_mappers.items()},
                        timeslice_feat_dim=timeslice_input_dim_gnn,
                        concept_embedding_dim=gnn_config.get('concept_embedding_dim', 64),
                        gnn_hidden_dim=gnn_config.get('gnn_hidden_dim', 128),
                        gnn_output_dim=gnn_config.get('gnn_output_dim', 128),  # Output of GNN layers before final FC
                        num_gnn_layers=gnn_config.get('num_gnn_layers', 2),
                        num_gat_heads=gnn_config.get('num_gat_heads', 4),
                        output_classes=1 if num_classes == 2 else num_classes,  # BCEWithLogitsLoss if binary
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
                            gnn_outputs = gnn_model_inner(gnn_batch)  # HeteroData batch
                            gnn_loss = gnn_criterion_inner(gnn_outputs,
                                                           gnn_batch.y.float())  # Ensure y is float for BCE
                            gnn_loss.backward()
                            gnn_optimizer_inner.step()
                            epoch_loss_gnn_sum += gnn_loss.item()
                        logger.debug(
                            f"Inner Fold {inner_fold_idx + 1}, GNN Epoch {epoch + 1}/{epochs_gnn_inner}, Avg Train Loss: {epoch_loss_gnn_sum / len(train_gnn_loader):.4f}")

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
                            gnn_oof_probas_2d = np.hstack(
                                [1 - concatenated_preds_gnn.reshape(-1, 1), concatenated_preds_gnn.reshape(-1, 1)])
                        else:  # Multiclass: ensure shape is (N, num_classes) - GNN output needs adjustment if not
                            gnn_oof_probas_2d = concatenated_preds_gnn  # This assumes GNN outputs (N,num_classes) for multiclass

                        if oof_preds_inner['gnn'][inner_val_idx].shape == gnn_oof_probas_2d.shape:
                            oof_preds_inner['gnn'][inner_val_idx] = gnn_oof_probas_2d
                        else:
                            logger.error(
                                f"GNN OOF shape mismatch. Target: {oof_preds_inner['gnn'][inner_val_idx].shape}, Value: {gnn_oof_probas_2d.shape}")
                            # Fill with default if mismatch
                            oof_preds_inner['gnn'][inner_val_idx] = np.full((len(inner_val_idx), num_classes),
                                                                            1 / num_classes)

                    # --- GNN Outer Test Predictions (from this inner fold's GNN model) ---
                    # This requires outer_test_graph_dataset to be ready
                    if outer_test_graph_dataset and len(outer_test_graph_dataset) > 0:
                        outer_test_gnn_loader = PyGDataLoader(outer_test_graph_dataset, batch_size=gnn_batch_size,
                                                              shuffle=False)
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
                                gnn_test_probas_2d = np.hstack([1 - concatenated_test_preds_gnn.reshape(-1, 1),
                                                                concatenated_test_preds_gnn.reshape(-1, 1)])
                            else:
                                gnn_test_probas_2d = concatenated_test_preds_gnn  # Adjust if multiclass output from GNN is different

                            if base_model_preds_on_outer_test_sum['gnn'].shape == gnn_test_probas_2d.shape:
                                base_model_preds_on_outer_test_sum['gnn'] += gnn_test_probas_2d / n_inner_folds
                            else:
                                logger.error(
                                    f"GNN Test Sum shape mismatch. Target: {base_model_preds_on_outer_test_sum['gnn'].shape}, Value: {gnn_test_probas_2d.shape}")

                    logger.info(f"Inner Fold {inner_fold_idx + 1}: HeteroTemporalGNN training and prediction complete.")

                except Exception as e_gnn_inner:
                    logger.error(f"Inner Fold {inner_fold_idx + 1}: Error during HeteroTemporalGNN: {e_gnn_inner}")
                    import traceback
                    logger.error(f"GNN Traceback: {traceback.format_exc()}")
                    # Fill with defaults if GNN fails for this inner fold
                    oof_preds_inner['gnn'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1 / num_classes)
                    base_model_preds_on_outer_test_sum['gnn'] += np.full((len(y_outer_test), num_classes),
                                                                         1 / num_classes) / n_inner_folds
            # --- End GNN Block ---

        # --- Length of Stay (LoS) Regression ---
        los_column_name = 'lengthofStay'  # Make sure this matches the actual column name
        predicted_los_outer_test = np.full(len(y_outer_test), np.nan)  # Default to NaN

        if los_column_name in X_outer_train_raw_fold_df.columns and los_column_name in X_outer_test_raw_fold_df.columns:
            y_los_outer_train_raw = X_outer_train_raw_fold_df[los_column_name].values
            y_los_outer_test_actual_orig_scale = X_outer_test_raw_fold_df[
                los_column_name].values  # Keep original scale for final MAE

            # Ensure no NaN values in target for LoS training
            valid_los_train_indices = ~np.isnan(y_los_outer_train_raw)

            if np.sum(valid_los_train_indices) > 0:
                X_los_train_fold_processed = X_outer_train_processed[valid_los_train_indices]
                y_los_outer_train_cleaned_orig_scale = y_los_outer_train_raw[valid_los_train_indices]

                # Apply log1p transformation to the LoS target for training
                y_los_outer_train_transformed = np.log1p(y_los_outer_train_cleaned_orig_scale)

                logger.info(
                    f"Outer Fold {outer_fold_idx + 1}: Training Quantile LightGBM Regressor for Length of Stay (log1p transformed target)...")
                try:
                    lgbm_los_config = config.get('ensemble', {}).get('lgbm_los_params', {})

                    los_regressor_params = {
                        'objective': 'quantile',  # Quantile regression
                        'alpha': 0.5,  # For median (L1 loss equivalent for quantile)
                        'metric': 'quantile',  # Use quantile metric
                        'n_estimators': lgbm_los_config.get('n_estimators', 200),
                        'learning_rate': lgbm_los_config.get('learning_rate', 0.05),
                        'num_leaves': lgbm_los_config.get('num_leaves', 31),
                        'max_depth': lgbm_los_config.get('max_depth', -1),
                        'min_child_samples': lgbm_los_config.get('min_child_samples', 20),
                        'subsample': lgbm_los_config.get('subsample', 0.8),
                        'colsample_bytree': lgbm_los_config.get('colsample_bytree', 0.8),
                        'random_state': seed + outer_fold_idx,
                        'n_jobs': -1,
                        'verbose': -1,
                    }
                    los_model = LightGBMModel(params=los_regressor_params)

                    optuna_los_config = config.get('optuna', {}).get('lgbm_los', {})
                    use_optuna_for_los = optuna_los_config.get('use_optuna_for_los', False)  # Control flag
                    n_trials_los = optuna_los_config.get('n_trials', 15)
                    timeout_los = optuna_los_config.get('timeout_seconds_per_fold', 300)

                    if use_optuna_for_los and X_los_train_fold_processed.shape[0] > 20:  # Ensure enough data for split
                        # Create a train/validation split from X_los_train_fold_processed for Optuna
                        # Using a simple split here, can be improved (e.g., KFold for robustness within Optuna)
                        # For simplicity, let's use a fixed 80/20 split of the already cleaned LoS training data
                        # Ensure y_los_outer_train_cleaned is not all the same value before trying StratifiedShuffleSplit
                        # For regression, a simple train_test_split is fine.
                        from sklearn.model_selection import train_test_split as los_optuna_split

                        X_optuna_los_train, X_optuna_los_val, \
                            y_optuna_los_train_transformed, y_optuna_los_val_transformed = los_optuna_split(
                            X_los_train_fold_processed, y_los_outer_train_transformed,
                            # Use transformed target for Optuna
                            test_size=0.25,
                            random_state=seed + outer_fold_idx + 100
                        )

                        def los_objective(trial):
                            trial_los_params = {
                                'objective': 'quantile',
                                'alpha': 0.5,
                                'metric': 'quantile',
                                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                                'num_leaves': trial.suggest_int('num_leaves', 10, 100, log=True),
                                'max_depth': trial.suggest_int('max_depth', 3, 10),
                                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                                'random_state': seed + outer_fold_idx + trial.number,
                                'n_jobs': -1,
                                'verbose': -1,
                            }
                            temp_los_model = LightGBMModel(params=trial_los_params)
                            temp_los_model.train(  # Train on transformed data
                                X_optuna_los_train, y_optuna_los_train_transformed,
                                X_optuna_los_val, y_optuna_los_val_transformed,
                                early_stopping_rounds=optuna_los_config.get('early_stopping_rounds_trial', 10)
                            )
                            preds_val_los_transformed = temp_los_model.predict(X_optuna_los_val)
                            preds_val_los_orig_scale = np.expm1(preds_val_los_transformed)
                            # Optuna needs to evaluate on original scale MAE if possible, or transformed MAE
                            # For simplicity, let's use MAE on transformed scale for Optuna, as it's simpler for ES.
                            # The final model will be evaluated on original scale.
                            # Alternatively, transform y_optuna_los_val_transformed back for MAE calculation here.
                            y_optuna_los_val_orig_scale = np.expm1(y_optuna_los_val_transformed)
                            mae_orig_scale = mean_absolute_error(y_optuna_los_val_orig_scale, preds_val_los_orig_scale)
                            return mae_orig_scale  # Optuna minimizes this

                        study_los = optuna.create_study(direction='minimize',
                                                        sampler=optuna.samplers.TPESampler(
                                                            seed=seed + outer_fold_idx + 200))
                        study_los.optimize(los_objective, n_trials=n_trials_los, timeout=timeout_los)

                        best_los_params_optuna = study_los.best_params
                        best_los_score_optuna = study_los.best_value  # This is MAE on original scale from Optuna
                        logger.info(
                            f"Outer Fold {outer_fold_idx + 1}: Best LoS Regressor params from Optuna: {best_los_params_optuna}, Best MAE (Optuna eval): {best_los_score_optuna:.4f}")
                        wandb.log({
                            f"outer_fold_{outer_fold_idx + 1}/los_optuna_best_params": best_los_params_optuna,
                            f"outer_fold_{outer_fold_idx + 1}/los_optuna_best_mae": best_los_score_optuna
                        })

                        final_los_params = los_regressor_params.copy()
                        final_los_params.update(best_los_params_optuna)
                        final_los_params['objective'] = 'quantile'
                        final_los_params['alpha'] = 0.5
                        final_los_params['metric'] = 'quantile'

                        los_model = LightGBMModel(params=final_los_params)
                        los_model.train(X_los_train_fold_processed, y_los_outer_train_transformed,
                                        # Train on full transformed training data for the fold
                                        early_stopping_rounds=lgbm_los_config.get('early_stopping_rounds_final', 20))
                    else:
                        if use_optuna_for_los:
                            logger.warning(
                                f"Outer Fold {outer_fold_idx + 1}: Skipping Optuna for LoS due to insufficient data ({X_los_train_fold_processed.shape[0]} samples). Using base lgbm_los_params.")
                        los_model.train(X_los_train_fold_processed, y_los_outer_train_transformed,
                                        # Train on transformed
                                        early_stopping_rounds=lgbm_los_config.get('early_stopping_rounds_final', 20),
                                        )
                    predicted_los_transformed = los_model.predict(X_outer_test_processed)
                    logger.info(f"LoS predictions before np.expm1: {predicted_los_transformed[:5]}")
                    predicted_los_outer_test = np.expm1(predicted_los_transformed)
                    logger.info(f"LoS predictions after np.expm1: {predicted_los_outer_test[:5]}")
                    predicted_los_outer_test = np.maximum(0,
                                                          predicted_los_outer_test)  # Ensure non-negativity after transform

                    valid_los_indices_for_mae = ~np.isnan(y_los_outer_test_actual_orig_scale) & ~np.isnan(
                        predicted_los_outer_test)
                    if np.sum(valid_los_indices_for_mae) > 0:
                        mae_los_outer = mean_absolute_error(
                            y_los_outer_test_actual_orig_scale[valid_los_indices_for_mae],
                            predicted_los_outer_test[valid_los_indices_for_mae]
                        )
                    else:
                        logger.warning(
                            f"Outer Fold {outer_fold_idx + 1}: No valid (non-NaN) actual/predicted LoS pairs for MAE calculation. MAE set to NaN.")
                        mae_los_outer = np.nan

                    if np.isnan(mae_los_outer):
                        logger.warning(f"Outer Fold {outer_fold_idx + 1}: MAE for LoS is NaN. Setting LSscore to 10.")
                        ls_score_outer = ls_score_calc(np.nan)
                    else:
                        ls_score_outer = ls_score_calc(mae_los_outer)

                    outer_fold_los_metrics['mae_los'].append(mae_los_outer)
                    outer_fold_los_metrics['ls_score'].append(ls_score_outer)
                    wandb.log({
                        f"outer_fold_{outer_fold_idx + 1}/ls_score": ls_score_outer,
                        f"outer_fold_{outer_fold_idx + 1}/mae_los": mae_los_outer,
                        "outer_fold": outer_fold_idx + 1
                    })
                    logger.info(
                        f"Outer Fold {outer_fold_idx + 1} LoS Regressor: MAE={mae_los_outer:.4f}, LSscore={ls_score_outer:.4f}")

                except Exception as e_los:
                    logger.error(f"Outer Fold {outer_fold_idx + 1}: Error during LoS Regression: {e_los}")
                    outer_fold_los_metrics['mae_los'].append(np.nan)
                    outer_fold_los_metrics['ls_score'].append(10.0)  # Worst score
            else:
                logger.warning(
                    f"Outer Fold {outer_fold_idx + 1}: No valid LoS data to train LoS regressor after NaN removal. LSscore will be 10.")
                outer_fold_los_metrics['mae_los'].append(np.nan)
                outer_fold_los_metrics['ls_score'].append(10.0)
        else:
            logger.warning(
                f"Outer Fold {outer_fold_idx + 1}: '{los_column_name}' not found in raw data. Skipping LoS regression. LSscore will be 10.")
            outer_fold_los_metrics['mae_los'].append(np.nan)
            outer_fold_los_metrics['ls_score'].append(10.0)

        all_predicted_los_list.append(predicted_los_outer_test)
        # --- End Length of Stay (LoS) Regression ---

        logger.info(f"Outer Fold {outer_fold_idx + 1}: Finished generating OOF predictions from inner CV.")
        meta_features_train_outer_list = []
        if config.get('ensemble', {}).get('train_lgbm', True): meta_features_train_outer_list.append(
            oof_preds_inner['lgbm'])
        if config.get('ensemble', {}).get('train_teco', True): meta_features_train_outer_list.append(
            oof_preds_inner['teco'])
        if train_gnn:  # Add GNN OOF preds if GNN was trained
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
                optuna_meta_config = config.get('optuna', {}).get('xgboost_meta', {})

                xgb_meta_model_outer = XGBoostMetaLearner(
                    params=meta_config.get('model_specific_params'),  # Initial base params
                    depth=meta_config.get('depth', 3)  # Default depth, Optuna can override if 'max_depth' is tuned
                )

                # Pass X_outer_test_processed and y_outer_test for validation if Optuna needs it
                # However, XGBoostMetaLearner's train method expects X_val, y_val to be from the *training set* for Optuna,
                # which is X_meta_train_outer and y_meta_train_outer itself if we want to use a validation split from that.
                # For meta-learner, typically we train on all OOF preds. If Optuna needs a val set,
                # it should split X_meta_train_outer. The current XGBoostMetaLearner setup implies
                # X_val, y_val passed to its train are for *that specific training run*.
                # Let's assume for now that if Optuna is used, it needs a validation set derived from X_meta_train_outer.
                # The XGBoostMetaLearner current Optuna objective uses the evals_list, which can include a validation set.
                # We will pass a portion of X_meta_train_outer as validation for Optuna if enabled.

                X_meta_val_optuna, y_meta_val_optuna = None, None
                if optuna_meta_config.get('use_optuna_for_meta', False) and X_meta_train_outer.shape[
                    0] > 10:  # Basic check
                    # Simple split for Optuna validation - can be improved (e.g. stratified)
                    # This split is ONLY for Optuna's internal validation during hyperparameter search.
                    # The final model will be trained on the full X_meta_train_outer.
                    # Note: The XGBoostMetaLearner.train() method itself doesn't do this split.
                    # It expects X_val, y_val. We need to decide if we pass a split here, or if Optuna
                    # runs CV internally (which it can). The current setup of XGBoostPruningCallback
                    # uses an 'eval' set.
                    # For simplicity, we'll rely on XGBoost's early stopping with X_val if provided.
                    # Optuna will use this X_val for its objective.
                    # The meta-learner is trained on OOF predictions, so a further split might reduce data too much.
                    # The `train` method of `XGBoostMetaLearner` already accepts X_val, y_val.
                    # We should pass X_meta_train_outer and y_meta_train_outer as X_train, y_train
                    # and potentially NOT pass X_val, y_val to Optuna, letting it use its default (e.g. CV)
                    # or the early stopping within the trial based on the 'eval' set if only train is in evals_list.
                    # The current XGBoostMetaLearner's Optuna objective uses `evals_list` which can have `dval`.
                    # We will train the meta-learner on the full X_meta_train_outer and y_meta_train_outer.
                    # For Optuna, if X_val/y_val are required by its objective, they should be passed.
                    # The current XGBoostMetaLearner.train() uses X_val, y_val for early stopping in trials.
                    # We will *not* split X_meta_train_outer here. XGBoostMetaLearner's train will use
                    # its internal logic for validation if X_val, y_val are passed to it.
                    # The current meta-learner trains on all OOF data without a val split.
                    # So, for Optuna, we also won't provide a separate val split to its train method.
                    # It will use early stopping on the training data itself if no X_val is passed,
                    # or if X_val is passed, it will use that.
                    # The warning in XGBoostMetaLearner about X_val/y_val not provided for early stopping applies.
                    pass  # No explicit split here for X_meta_val_optuna

                from sklearn.model_selection import train_test_split

                X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
                    X_meta_train_outer, y_meta_train_outer, test_size=0.2, random_state=42
                )
                xgb_meta_model_outer.train(
                    X_meta_train, y_meta_train,
                    X_val=X_meta_val,
                    y_val=y_meta_val,
                    num_boost_round=meta_config.get('num_boost_round', 100),  # Updated value from new config
                    early_stopping_rounds=meta_config.get('early_stopping_rounds', 15),  # Updated value
                    use_optuna=optuna_meta_config.get('use_optuna_for_meta', False),
                    optuna_n_trials=optuna_meta_config.get('n_trials', 15),
                    optuna_timeout=optuna_meta_config.get('timeout_seconds_fold', 300),
                    wandb_run=wandb.run
                )
                logger.info(f"Outer Fold {outer_fold_idx + 1}: XGBoost Meta-Learner trained.")

                meta_features_test_outer_list = []
                if config.get('ensemble', {}).get('train_lgbm', True): meta_features_test_outer_list.append(
                    base_model_preds_on_outer_test_sum['lgbm'])
                if config.get('ensemble', {}).get('train_teco', True): meta_features_test_outer_list.append(
                    base_model_preds_on_outer_test_sum['teco'])
                if train_gnn:  # Add GNN test preds if GNN was trained
                    meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['gnn'])

                if not meta_features_test_outer_list:  # Should not happen if train_meta_learner is true and at least one base model ran
                    logger.error(
                        f"Outer Fold {outer_fold_idx + 1}: No base model predictions available for meta-learner test set. Skipping meta-learner prediction.")
                    # Fill meta metrics with NaN for this fold
                    for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)
                    wandb.log({f"outer_fold_{outer_fold_idx + 1}/meta_auroc": np.nan, "outer_fold": outer_fold_idx + 1})
                    # Skip to soft voting or next part of the loop for this outer fold
                    # This requires careful restructuring of the following soft-voting block or a continue
                else:
                    X_meta_test_outer = np.concatenate(meta_features_test_outer_list, axis=1)
                    final_preds_meta_proba = xgb_meta_model_outer.predict_proba(X_meta_test_outer)
                final_preds_meta_labels = xgb_meta_model_outer.predict(
                    X_meta_test_outer)  # Labels based on default 0.5 threshold from predict()

                # --- F1 Threshold Maximization ---
                death_label_value = class_mapping.get('Death', None)
                best_threshold_fold = 0.5  # Default
                f1_at_best_threshold = 0.0  # Default

                if death_label_value is not None and final_preds_meta_proba is not None:
                    logger.info(
                        f"Outer Fold {outer_fold_idx + 1}: Maximizing F1 threshold for 'Death' class (Meta-Learner)...")
                    best_threshold_fold, f1_at_best_threshold = maximise_f1_threshold(
                        y_true=y_outer_test,
                        y_probas=final_preds_meta_proba,
                        target_label_value=death_label_value,
                        class_mapping=class_mapping,
                        positive_label_name='Death'
                    )
                    thr_dict[f"meta_{outer_fold_idx}"] = best_threshold_fold
                    all_f1_at_best_thr_meta_list.append(f1_at_best_threshold)  # Store this F1
                    logger.info(
                        f"Outer Fold {outer_fold_idx + 1} Meta-Learner: Best F1 threshold for 'Death' = {best_threshold_fold:.4f} (yields F1 = {f1_at_best_threshold:.4f})")
                    wandb.log({
                        f"outer_fold_{outer_fold_idx + 1}/meta_best_f1_thr_death": best_threshold_fold,
                        f"outer_fold_{outer_fold_idx + 1}/meta_f1_at_best_thr_death": f1_at_best_threshold,
                        "outer_fold": outer_fold_idx + 1
                    })
                elif death_label_value is None:
                    logger.warning(
                        f"Outer Fold {outer_fold_idx + 1}: 'Death' class not in class_mapping. Cannot maximize F1 threshold. Using default 0.5.")
                    all_best_thresholds_fold_list.append(0.5)  # Append default if 'Death' class is missing
                else:  # final_preds_meta_proba is None
                    logger.warning(
                        f"Outer Fold {outer_fold_idx + 1}: Meta-learner probabilities are None. Cannot maximize F1 threshold. Using default 0.5.")
                    all_best_thresholds_fold_list.append(0.5)

                # Predictions using the new best_threshold_fold for metrics calculation
                if death_label_value is not None and final_preds_meta_proba is not None:
                    death_class_idx = class_mapping['Death']
                    final_preds_meta_labels_custom_thr = (
                                final_preds_meta_proba[:, death_class_idx] > best_threshold_fold).astype(int)
                    # Important: If 'Death' is class 0 and 'Survival' is 1, predictions should be 0 for Death, 1 for Survival.
                    # The line above gives 1 if P(Death) > thr, 0 otherwise. This needs to map back to original labels.
                    # If target_label_value for Death is 0, then:
                    #   predicted_as_death = (final_preds_meta_proba[:, death_class_idx] > best_threshold_fold)
                    #   final_preds_meta_labels_custom_thr = np.where(predicted_as_death, death_label_value, class_mapping.get('Survival'))
                    # This is complex. Simpler: use the f1_at_best_threshold directly for DTscore.
                    # For other metrics (acc, precision, recall), using labels derived from this custom threshold for 'Death'
                    # while other classes (if any) use argmax might be inconsistent.
                    # The user's request focuses on F1_Death and DTscore. Let's use f1_at_best_threshold for DTscore.
                    # And for overall metrics, we can report based on default 0.5 or this custom threshold.
                    # For now, let's keep final_preds_meta_labels (from default 0.5) for general metrics,
                    # and use f1_at_best_threshold for DTscore.
                # --- End F1 Threshold Maximization ---

                acc = accuracy_score(y_outer_test,
                                                final_preds_meta_labels)  # Based on default 0.5 threshold
                f1 = f1_score(y_outer_test, final_preds_meta_labels,  # Based on default 0.5 threshold
                                         average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                prec = precision_score(y_outer_test, final_preds_meta_labels,
                                                  # Based on default 0.5 threshold
                                                  average='weighted' if num_classes > 2 else 'binary',
                                                  zero_division=0)
                rec = recall_score(y_outer_test, final_preds_meta_labels,
                                              # Based on default 0.5 threshold
                                              average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                auroc = -1.0
                dt_score = 10.0  # Default to worst score

                try:
                    probas_for_auc = final_preds_meta_proba[:,
                                     1] if num_classes == 2 and final_preds_meta_proba.ndim == 2 and \
                                           final_preds_meta_proba.shape[
                                               1] >= 2 else final_preds_meta_proba
                    auroc = roc_auc_score(y_outer_test, probas_for_auc, multi_class='ovr',
                                                     average='weighted')
                except ValueError as e:
                    logger.warning(
                        f"Outer Fold {outer_fold_idx + 1} Meta AUROC calc error: {e}. Proba shape: {final_preds_meta_proba.shape if isinstance(final_preds_meta_proba, np.ndarray) else 'N/A'}")

                # Calculate DTscore for Meta-Learner
                # Uses f1_at_best_threshold if available, otherwise falls back to F1 from default threshold predictions
                f1_death_for_dtscore = 0.0
                if death_label_value is not None:
                    if f1_at_best_threshold > 0.0:  # Check if maximization was successful
                        f1_death_for_dtscore = f1_at_best_threshold
                        logger.info(
                            f"Outer Fold {outer_fold_idx + 1} Meta-Learner: Using F1_Death from maximized threshold ({f1_death_for_dtscore:.4f}) for DTscore.")
                    else:  # Fallback to F1 from default 0.5 threshold if maximization yielded 0 or was skipped
                        f1_death_for_dtscore = f1_score(y_outer_test, final_preds_meta_labels,
                                                             # Labels from default 0.5
                                                             labels=[death_label_value], pos_label=death_label_value,
                                                             average='binary', zero_division=0)
                        logger.info(
                            f"Outer Fold {outer_fold_idx + 1} Meta-Learner: Using F1_Death from default 0.5 threshold ({f1_death_for_dtscore:.4f}) for DTscore.")
                    dt_score = dt_score_calc(f1_death_for_dtscore)
                else:
                    logger.warning(
                        f"Outer Fold {outer_fold_idx + 1} Meta-Learner: 'Death' class not found in class_mapping. DTscore set to 10.")
                    dt_score = dt_score_calc(np.nan)  # Ensure it's 10

                outer_fold_metrics_meta['accuracy'].append(acc)
                outer_fold_metrics_meta['auroc'].append(auroc)
                outer_fold_metrics_meta['f1'].append(f1)  # This is overall F1
                outer_fold_metrics_meta['precision'].append(prec)
                outer_fold_metrics_meta['recall'].append(rec)
                outer_fold_metrics_meta['dt_score'].append(dt_score)  # Based on best/maximized F1 for Death

                # Calculate GLscore for Meta-Learner
                current_ls_score = outer_fold_los_metrics['ls_score'][-1] if outer_fold_los_metrics[
                    'ls_score'] else ls_score_calc(np.nan)  # Default to worst if no LS score
                gl_score = gl_score_calc(dt_score, current_ls_score)
                outer_fold_metrics_meta['gl_score'].append(gl_score)

                wandb.log({
                    f"outer_fold_{outer_fold_idx + 1}/meta_f1_death_for_dtscore": f1_death_for_dtscore if death_label_value is not None else np.nan,
                    # Log the original F1_Death based on 0.5 threshold for comparison if needed
                    f"outer_fold_{outer_fold_idx + 1}/meta_f1_death_default_thr": f1_score(y_outer_test,
                                                                                           final_preds_meta_labels,
                                                                                           labels=[death_label_value],
                                                                                           pos_label=death_label_value,
                                                                                           average='binary',
                                                                                           zero_division=0) if death_label_value is not None else np.nan,
                    f"outer_fold_{outer_fold_idx + 1}/meta_mae_los": outer_fold_los_metrics['mae_los'][-1] if
                    outer_fold_los_metrics['mae_los'] else np.nan,
                    f"outer_fold_{outer_fold_idx + 1}/meta_dt_score": dt_score,
                    f"outer_fold_{outer_fold_idx + 1}/meta_ls_score": current_ls_score,
                    f"outer_fold_{outer_fold_idx + 1}/meta_gl_score": gl_score,
                    f"outer_fold_{outer_fold_idx + 1}/meta_auroc": auroc,
                    f"outer_fold_{outer_fold_idx + 1}/meta_acc": acc,  # Based on default 0.5 thr
                    "outer_fold": outer_fold_idx + 1
                })
                logger.info(
                    f"Outer Fold {outer_fold_idx + 1} Meta-Learner: "
                    f"F1_Death(for DTscore)={f1_death_for_dtscore if death_label_value is not None else np.nan:.4f} (BestThr={best_threshold_fold:.2f}), "
                    f"MAE_LoS={outer_fold_los_metrics['mae_los'][-1] if outer_fold_los_metrics['mae_los'] else np.nan:.4f}, "
                    f"DTscore={dt_score:.4f}, LSscore={current_ls_score:.4f}, GLscore={gl_score:.4f} | "
                    f"AUROC={auroc:.4f}, Acc(0.5 thr)={acc:.4f}"
                )

                # --- Accumulate predictions for CSV logging ---
                all_test_indices_list.append(outer_test_idx)
                all_y_test_list.append(y_outer_test)  # Actual outcomeType
                # Store probabilities for 'Death' class (class 0) for later thresholding if needed for CSV
                # final_preds_meta_proba is (N, num_classes), class_mapping['Death'] is its index
                if final_preds_meta_proba is not None and class_mapping.get('Death') is not None:
                    probas_death = final_preds_meta_proba[:, class_mapping['Death']]
                    all_probas_list.append(probas_death)
                else:  # Fallback if probas are not available, store NaNs or re-derive from labels if only labels are available
                    all_probas_list.append(np.full(len(y_outer_test), np.nan))

                all_preds_list.append(
                    final_preds_meta_labels)  # Store labels derived from default threshold for metrics

                # Get actual lengthOfStay for these test indices from the original X_full_raw_df
                # Ensure 'lengthofStay' is a valid column name. From logs, it seems to be.
                los_column_name = 'lengthofStay'  # As seen in logs
                if los_column_name in X_full_raw_df.columns:
                    actual_los_fold = X_full_raw_df.iloc[outer_test_idx][los_column_name].values
                    all_actual_los_list.append(actual_los_fold)
                else:
                    logger.warning(
                        f"Column '{los_column_name}' not found in X_full_raw_df. Cannot log actual Length of Stay.")
                    all_actual_los_list.append(np.full(len(outer_test_idx), np.nan))  # Append NaNs if not found

                # Capture patient IDs if patient_id_col_name_for_gnn is set and valid
                if patient_id_col_name_for_gnn and patient_id_col_name_for_gnn in X_full_raw_df.columns:
                    patient_ids_fold = X_full_raw_df.iloc[outer_test_idx][patient_id_col_name_for_gnn].values
                    all_patient_ids_list.append(patient_ids_fold)
                elif X_full_raw_df.index.name == patient_id_col_name_for_gnn:  # If it was the index
                    patient_ids_fold = X_full_raw_df.iloc[outer_test_idx].index.values
                    all_patient_ids_list.append(patient_ids_fold)
                else:  # Fallback to original index if no specific patient ID column
                    patient_ids_fold = X_full_raw_df.iloc[outer_test_idx].index.values
                    all_patient_ids_list.append(patient_ids_fold)
                # --- End Accumulate predictions ---

            except Exception as e:
                logger.error(f"Outer Fold {outer_fold_idx + 1}: Error during Meta-Learner: {e}")
                for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)
                wandb.log({f"outer_fold_{outer_fold_idx + 1}/meta_auroc": np.nan, "outer_fold": outer_fold_idx + 1})
                # Also append NaNs or empty arrays to tracking lists if meta-learner fails for a fold
                all_test_indices_list.append(outer_test_idx)  # Still log indices
                all_y_test_list.append(y_outer_test)
                all_preds_meta_list.append(np.full(len(y_outer_test), np.nan))  # NaN for predictions
                if 'lengthofStay' in X_full_raw_df.columns:
                    all_actual_los_list.append(X_full_raw_df.iloc[outer_test_idx]['lengthofStay'].values)
                else:
                    all_actual_los_list.append(np.full(len(y_outer_test), np.nan))
                if patient_id_col_name_for_gnn and patient_id_col_name_for_gnn in X_full_raw_df.columns:
                    all_patient_ids_list.append(X_full_raw_df.iloc[outer_test_idx][patient_id_col_name_for_gnn].values)
                else:
                    all_patient_ids_list.append(X_full_raw_df.iloc[outer_test_idx].index.values)

        soft_vote_weights = config.get('ensemble', {}).get('soft_vote_weights', {})

        # Determine active models for soft voting, now potentially including GNN
        potential_sv_models = ['lgbm', 'teco']
        if train_gnn:
            potential_sv_models.append('gnn')

        active_models_for_sv = [model_key for model_key in potential_sv_models
                                if config.get('ensemble', {}).get(f'train_{model_key}', True) and \
                                model_key in soft_vote_weights and \
                                model_key in base_model_preds_on_outer_test_sum  # Ensure predictions exist
                                ]

        if soft_vote_weights and active_models_for_sv:
            logger.info(
                f"Outer Fold {outer_fold_idx + 1}: Performing Soft Voting with models: {active_models_for_sv}...")
            try:
                final_preds_soft_vote_proba_outer = np.zeros((len(y_outer_test), num_classes))
                total_weight = 0.0

                for model_key in active_models_for_sv:  # Iterate only over active, weighted models
                    weight = soft_vote_weights.get(model_key, 0)  # Should be >0 due to check above
                    # Ensure the predictions are valid before adding
                    preds_to_add = base_model_preds_on_outer_test_sum.get(model_key)

                    if preds_to_add is not None and preds_to_add.shape == final_preds_soft_vote_proba_outer.shape:
                        final_preds_soft_vote_proba_outer += weight * preds_to_add
                        total_weight += weight
                    else:
                        logger.warning(
                            f"Soft Voting: Skipping model {model_key} due to invalid predictions (shape {preds_to_add.shape if preds_to_add is not None else 'None'} vs target {final_preds_soft_vote_proba_outer.shape} or missing).")

                if total_weight > 1e-6:  # Check if any valid weighted predictions were actually added
                    # Normalize probabilities if total_weight is not 1 (or close to it)
                    # This is more robust if weights are relative importance rather than summing to 1.
                    final_preds_soft_vote_proba_outer /= total_weight
                    # Ensure probabilities sum to 1 per sample after weighting and summing
                    row_sums = final_preds_soft_vote_proba_outer.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    final_preds_soft_vote_proba_outer = final_preds_soft_vote_proba_outer / row_sums

                    final_preds_soft_vote_labels_outer = np.argmax(final_preds_soft_vote_proba_outer, axis=1)

                    best_threshold_fold_sv, f1_at_best_threshold_sv = maximise_f1_threshold(
                        y_true=y_outer_test,
                        y_probas=final_preds_soft_vote_proba_outer,
                        target_label_value=death_label_value,
                        class_mapping=class_mapping,
                        positive_label_name='Death'
                    )
                    thr_dict[f"sv_{outer_fold_idx}"] = best_threshold_fold_sv
                    logger.info(
                        f"Outer Fold {outer_fold_idx + 1} Soft Vote: Best F1 threshold for 'Death' = {best_threshold_fold_sv:.4f} (yields F1 = {f1_at_best_threshold_sv:.4f})")
                    wandb.log({
                        f"outer_fold_{outer_fold_idx + 1}/sv_best_f1_thr_death": best_threshold_fold_sv,
                        f"outer_fold_{outer_fold_idx + 1}/sv_f1_at_best_thr_death": f1_at_best_threshold_sv,
                        "outer_fold": outer_fold_idx + 1
                    })

                    death_class_idx = class_mapping['Death']
                    survival_class_idx = class_mapping.get('Survival', 1 - death_class_idx)
                    final_preds_soft_vote_labels_outer = np.where(final_preds_soft_vote_proba_outer[:, death_class_idx] > best_threshold_fold_sv, death_class_idx, survival_class_idx)

                    acc_sv_outer = accuracy_score(y_outer_test, final_preds_soft_vote_labels_outer)
                    f1_sv_outer = f1_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                           average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    prec_sv_outer = precision_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                                    average='weighted' if num_classes > 2 else 'binary',
                                                    zero_division=0)
                    rec_sv_outer = recall_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                                average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    auroc_sv_outer = -1.0
                    dt_score_sv_outer = 10.0  # Default to worst score

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

                    # Calculate DTscore for Soft Voting
                    death_label_value = class_mapping.get('Death', None)
                    if death_label_value is not None:
                        f1_death_sv = f1_score(y_outer_test, final_preds_soft_vote_labels_outer,
                                               labels=[death_label_value], pos_label=death_label_value,
                                               average='binary', zero_division=0)
                        dt_score_sv_outer = dt_score_calc(f1_death_sv)  # Use function
                        logger.info(
                            f"Outer Fold {outer_fold_idx + 1} Soft Vote: F1_Death={f1_death_sv:.4f}, DTscore={dt_score_sv_outer:.4f}")
                    else:
                        logger.warning(
                            f"Outer Fold {outer_fold_idx + 1} Soft Vote: 'Death' class not found in class_mapping. DTscore set to 10.")
                        dt_score_sv_outer = dt_score_calc(np.nan)  # Ensure it's 10 if class not found

                    outer_fold_metrics_soft_vote['accuracy'].append(acc_sv_outer)
                    outer_fold_metrics_soft_vote['auroc'].append(auroc_sv_outer)
                    outer_fold_metrics_soft_vote['f1'].append(f1_sv_outer)
                    outer_fold_metrics_soft_vote['precision'].append(prec_sv_outer)
                    outer_fold_metrics_soft_vote['recall'].append(rec_sv_outer)
                    outer_fold_metrics_soft_vote['dt_score'].append(dt_score_sv_outer)

                    # Calculate GLscore for Soft Voting
                    current_ls_score_sv = outer_fold_los_metrics['ls_score'][-1] if outer_fold_los_metrics[
                        'ls_score'] else ls_score_calc(np.nan)  # Default to worst if no LS score
                    gl_score_sv_outer = gl_score_calc(dt_score_sv_outer, current_ls_score_sv)
                    outer_fold_metrics_soft_vote['gl_score'].append(gl_score_sv_outer)

                    wandb.log({
                        f"outer_fold_{outer_fold_idx + 1}/sv_f1_death": f1_death_sv if death_label_value is not None else np.nan,
                        f"outer_fold_{outer_fold_idx + 1}/sv_mae_los": outer_fold_los_metrics['mae_los'][-1] if
                        outer_fold_los_metrics['mae_los'] else np.nan,
                        f"outer_fold_{outer_fold_idx + 1}/sv_dt_score": dt_score_sv_outer,
                        f"outer_fold_{outer_fold_idx + 1}/sv_ls_score": current_ls_score_sv,
                        f"outer_fold_{outer_fold_idx + 1}/sv_gl_score": gl_score_sv_outer,
                        f"outer_fold_{outer_fold_idx + 1}/sv_auroc": auroc_sv_outer,
                        f"outer_fold_{outer_fold_idx + 1}/sv_acc": acc_sv_outer,
                        "outer_fold": outer_fold_idx + 1
                    })
                    logger.info(
                        f"Outer Fold {outer_fold_idx + 1} Soft Vote: "
                        f"F1_Death={f1_death_sv if death_label_value is not None else np.nan:.4f}, "
                        f"MAE_LoS={outer_fold_los_metrics['mae_los'][-1] if outer_fold_los_metrics['mae_los'] else np.nan:.4f}, "
                        f"DTscore={dt_score_sv_outer:.4f}, LSscore={current_ls_score_sv:.4f}, GLscore={gl_score_sv_outer:.4f} | "
                        f"AUROC={auroc_sv_outer:.4f}, Acc={acc_sv_outer:.4f}"
                    )
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
        for metric_name_meta, values in outer_fold_metrics_meta.items():  # Iterate over all including dt_score
            if metric_name_meta not in ['auroc']:  # Exclude auroc as it's already logged with std
                avg_val = np.nanmean(values)
                wandb.summary[f"ncv_meta_avg_{metric_name_meta}"] = avg_val
                logger.info(f"Meta-Learner Average {metric_name_meta.replace('_', ' ').capitalize()}: {avg_val:.4f}")
        # Detailed GL Score printout for Meta-Learner
        avg_meta_gl_score = np.nanmean(outer_fold_metrics_meta['gl_score'])
        avg_meta_dt_score = np.nanmean(outer_fold_metrics_meta['dt_score'])
        # avg_ls_score needs to be defined before this point, or use np.nanmean(outer_fold_los_metrics['ls_score'])
        # Assuming avg_ls_score will be available from the LoS summary section that comes later,
        # or computed just before this summary block. For now, let's compute it here if not available.
        if 'avg_ls_score' not in locals() and 'avg_ls_score' not in globals():  # Check if avg_ls_score is already computed
            current_avg_ls_score = np.nanmean(outer_fold_los_metrics['ls_score']) if outer_fold_los_metrics[
                'ls_score'] else np.nan
        else:
            current_avg_ls_score = avg_ls_score  # Use existing if available

        if not np.isnan(avg_meta_gl_score):  # Only print if GL score is valid
            logger.info(
                f"Meta-Learner Average GLscore: {avg_meta_gl_score:.4f} (Avg DTscore={avg_meta_dt_score:.4f} + Avg LSscore={current_avg_ls_score:.4f})")

        if all_f1_at_best_thr_meta_list:
            avg_f1_death_optimized = np.nanmean([f1 for f1 in all_f1_at_best_thr_meta_list if not np.isnan(f1)])
            if not np.isnan(avg_f1_death_optimized):
                wandb.summary["ncv_meta_avg_f1_death_optimized"] = avg_f1_death_optimized
                logger.info(f"Meta-Learner Average F1_Death (at best threshold): {avg_f1_death_optimized:.4f}")
            else:
                wandb.summary["ncv_meta_avg_f1_death_optimized"] = np.nan
        else:
            wandb.summary["ncv_meta_avg_f1_death_optimized"] = np.nan


    else:
        logger.info("Meta-Learner metrics not computed or all NaN.")
        # Log NaN for all expected summary metrics if meta-learner didn't run or all were NaN
        for metric_name_meta in ['auroc', 'accuracy', 'f1', 'precision', 'recall', 'dt_score', 'gl_score']:
            wandb.summary[f"ncv_meta_avg_{metric_name_meta}"] = np.nan
            if metric_name_meta == 'auroc':  # Specific handling for std if auroc is NaN
                wandb.summary["ncv_meta_std_auroc"] = np.nan

    if soft_vote_weights and any(not np.isnan(v) for v in outer_fold_metrics_soft_vote['auroc']):
        avg_sv_auroc = np.nanmean(outer_fold_metrics_soft_vote['auroc'])
        std_sv_auroc = np.nanstd(outer_fold_metrics_soft_vote['auroc'])
        logger.info(f"Soft Voting Average AUROC: {avg_sv_auroc:.4f} +/- {std_sv_auroc:.4f}")
        wandb.summary["ncv_sv_avg_auroc"] = avg_sv_auroc
        wandb.summary["ncv_sv_std_auroc"] = std_sv_auroc
        for metric_name_sv, values in outer_fold_metrics_soft_vote.items():  # Iterate over all including dt_score
            if metric_name_sv not in ['auroc']:  # Exclude auroc as it's already logged with std
                avg_val = np.nanmean(values)
                wandb.summary[f"ncv_sv_avg_{metric_name_sv}"] = avg_val
                logger.info(f"Soft Voting Average {metric_name_sv.replace('_', ' ').capitalize()}: {avg_val:.4f}")
        # Detailed GL Score printout for Soft Voting
        avg_sv_gl_score = np.nanmean(outer_fold_metrics_soft_vote['gl_score'])
        avg_sv_dt_score = np.nanmean(outer_fold_metrics_soft_vote['dt_score'])
        # Assuming avg_ls_score is available from LoS summary or computed earlier
        if 'avg_ls_score' not in locals() and 'avg_ls_score' not in globals():
            current_avg_ls_score_sv = np.nanmean(outer_fold_los_metrics['ls_score']) if outer_fold_los_metrics[
                'ls_score'] else np.nan
        else:
            current_avg_ls_score_sv = avg_ls_score

        if not np.isnan(avg_sv_gl_score):  # Only print if GL score is valid
            logger.info(
                f"Soft Voting Average GLscore: {avg_sv_gl_score:.4f} (Avg DTscore={avg_sv_dt_score:.4f} + Avg LSscore={current_avg_ls_score_sv:.4f})")
    else:
        logger.info("Soft Voting metrics not computed or all NaN.")
        # Log NaN for all expected summary metrics if soft voting didn't run or all were NaN
        for metric_name_sv in ['auroc', 'accuracy', 'f1', 'precision', 'recall', 'dt_score', 'gl_score']:
            wandb.summary[f"ncv_sv_avg_{metric_name_sv}"] = np.nan
            if metric_name_sv == 'auroc':  # Specific handling for std if auroc is NaN
                wandb.summary["ncv_sv_std_auroc"] = np.nan

    # --- Log LoS NCV Summary ---
    if any(not np.isnan(v) for v in outer_fold_los_metrics['ls_score']):
        avg_ls_score = np.nanmean(outer_fold_los_metrics['ls_score'])
        avg_mae_los = np.nanmean(outer_fold_los_metrics['mae_los'])
        logger.info(f"Length of Stay Regressor Average LSscore: {avg_ls_score:.4f}")
        logger.info(f"Length of Stay Regressor Average MAE: {avg_mae_los:.4f}")
        wandb.summary["ncv_los_avg_ls_score"] = avg_ls_score
        wandb.summary["ncv_los_avg_mae"] = avg_mae_los
    else:
        logger.info("LoS metrics (LSscore, MAE) not computed or all NaN.")
        wandb.summary["ncv_los_avg_ls_score"] = np.nan
        wandb.summary["ncv_los_avg_mae"] = np.nan

    # --- Process and Log Accumulated Predictions ---
    if all_test_indices_list:
        try:
            logger.info("Processing accumulated predictions for CSV logging...")
            # Concatenate results from all folds
            all_indices_flat = np.concatenate(all_test_indices_list)
            all_y_test_flat = np.concatenate(all_y_test_list)
            # all_preds_meta_flat contains labels based on default threshold (used for metrics during CV)
            all_preds_meta_flat = np.concatenate(all_preds_meta_list)
            all_probas_death_meta_flat = np.concatenate(all_probas_meta_list)  # Probabilities for 'Death' class
            all_actual_los_flat = np.concatenate(all_actual_los_list)
            all_predicted_los_flat = np.concatenate(all_predicted_los_list)
            all_patient_ids_flat = np.concatenate(all_patient_ids_list)

            # Create a DataFrame for DTestimation.csv
            # Sort by original index to ensure consistent order if needed, though not strictly necessary for a simple vector.
            # df_dtest = pd.DataFrame({
            # 'original_index': all_indices_flat,
            # 'patient_id': all_patient_ids_flat, # patient_id or original_index
            # 'predicted_outcomeType': all_preds_meta_flat, # This is DTestimation
            # 'actual_outcomeType': all_y_test_flat
            # }).sort_values(by='original_index').reset_index(drop=True)

            # Simpler: just the prediction vector as requested.
            # We need to ensure the order is meaningful, e.g. sorted by patient_id or original index.
            # For now, let's create them based on concatenation order, assuming NCV folds are processed sequentially.
            # To ensure the vectors correspond to the original dataset order for the test samples:

            # Create a mapping from original index to prediction/actuals
            results_df = pd.DataFrame({
                'original_index': all_indices_flat,
                'patient_id': all_patient_ids_flat,
                'predicted_outcomeType': all_preds_meta_flat,
                'actual_lengthOfStay': all_actual_los_flat,  # Actuals for reference
                'predicted_lengthOfStay': all_predicted_los_flat,  # Predictions for LSscore
                'actual_outcomeType': all_y_test_flat
            })
            # Sort by the original index to restore order of samples as they appeared in X_full_raw_df
            results_df_sorted = results_df.sort_values(by='original_index').reset_index(drop=True)

            output_dir = config.get('output_dir', 'outputs')
            os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

            # DTestimation.csv: vector of predicted discharge type (outcomeType)
            # Apply the determined best F1 threshold.
            # Create a mapping from original index to fold index
            fold_map = {}
            for i in range(len(all_test_indices_list)):
                for j in all_test_indices_list[i]:
                    fold_map[j] = i

            # Sort by original index
            sorted_indices = np.sort(list(fold_map.keys()))

            # Create the prediction vector
            predicted_labels_for_csv = []
            for i in sorted_indices:
                fold_idx = fold_map[i]
                threshold = thr_dict.get(f"sv_{fold_idx}", 0.5)

                # Find the corresponding probability
                prob_death = all_probas_meta_list[fold_idx][np.where(all_test_indices_list[fold_idx] == i)[0][0]]

                death_val = class_mapping['Death']
                survival_val = [l for l in class_mapping.values() if l != death_val][0]

                if prob_death > threshold:
                    predicted_labels_for_csv.append(death_val)
                else:
                    predicted_labels_for_csv.append(survival_val)

            dt_estimation_df = pd.DataFrame({'predicted_outcome': predicted_labels_for_csv})
            dt_estimation_path = os.path.join(output_dir, "DTestimation.csv")
            dt_estimation_df.to_csv(dt_estimation_path, index=False, header=True)
            logger.info(f"Saved DTestimation.csv to {dt_estimation_path}")
            wandb.log_artifact(dt_estimation_path, name="DTestimation", type="predictions")

            # LSestimation.csv: vector of (predicted) length of stay
            # The requirement is "Estimated_LS", so we save the predicted values.
            ls_estimation_df = pd.DataFrame({'Estimated_LS': results_df_sorted['predicted_lengthOfStay']})
            ls_estimation_path = os.path.join(output_dir, "LSestimation.csv")
            ls_estimation_df.to_csv(ls_estimation_path, index=False, header=True)
            logger.info(f"Saved LSestimation.csv (with predicted LoS) to {ls_estimation_path}")
            wandb.log_artifact(ls_estimation_path, name="LSestimation", type="predictions")

            # Optionally, log a combined file for easier review
            combined_output_path = os.path.join(output_dir, "predictions_and_los_summary.csv")
            results_df_sorted.to_csv(combined_output_path, index=False)
            wandb.log_artifact(combined_output_path, name="full_test_set_predictions_summary",
                               type="predictions_summary")

        except Exception as e:
            logger.error(f"Error processing or logging accumulated prediction CSVs: {e}")
    else:
        logger.warning("No test predictions accumulated. Skipping CSV logging for DTestimation and LSestimation.")
    # --- End Process and Log ---

    # Save the thresholds dictionary
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    thresholds_path = os.path.join(output_dir, "thresholds.joblib")
    joblib.dump(thr_dict, thresholds_path)
    logger.info(f"Saved thresholds dictionary to {thresholds_path}")

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
