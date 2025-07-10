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
from models.stm_gnn import STMGNN # Assuming STMGNN is the main class
from data_utils.sequence_loader import TabularSequenceDataset, basic_collate_fn # For TECO
# For STM-GNN, data loading might be more complex (graph snapshots)
# from data_utils.graph_loader import GraphSnapshotDataset, graph_collate_fn # Conceptual

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
        tags=wandb_config.get('tags', ['full_run', 'ncv'])
    )
    logger.info(f"W&B initialized for project '{wandb_config.get('project', 'ifbme-project')}'")

    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu")
    logger.info(f"Using device: {device}")

    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    logger.info("Starting data loading and preprocessing for NCV...")
    use_dummy_data = config.get('use_dummy_data_for_full_run', False) # Control real vs dummy data

    if use_dummy_data:
        logger.info("Using dummy data generation for this run as per 'use_dummy_data_for_full_run' config.")
        num_total_samples = config.get('dummy_data_total_samples', 500)
        num_features = config.get('dummy_data_features', 20)
        num_classes_config = config.get('dummy_data_classes', 2)
        if num_classes_config < 2: raise ValueError("dummy_data_classes must be >= 2.")
        weights_combined = config.get('dummy_data_weights', [0.9, 0.1])
        p_combined = np.array(weights_combined) / np.sum(weights_combined)
        X_full_raw_df = pd.DataFrame(np.random.rand(num_total_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
        y_full_raw_series = pd.Series(np.random.choice(num_classes_config, num_total_samples, p=p_combined))
        logger.info(f"Dummy raw data generated: X_df shape {X_full_raw_df.shape}, y_series shape {y_full_raw_series.shape}")
    else:
        try:
            logger.info("Loading real data...")
            # load_raw_data now expects config and base_data_path
            # It should return X_full_df, y_full_series (or X_full_np, y_full_np)
            X_full_raw_df, y_full_raw_series = load_raw_data(config, base_data_path=config.get('data_dir', 'data/'))
            logger.info(f"Real data loaded: X_df shape {X_full_raw_df.shape}, y_series shape {y_full_raw_series.shape}")
        except Exception as e:
            logger.critical(f"Failed to load real data: {e}. Exiting.")
            sys.exit(1)

    unique_classes_full_raw = np.unique(y_full_raw_series)
    num_classes = len(unique_classes_full_raw)
    logger.info(f"Determined number of classes for NCV: {num_classes} from y_full_raw unique values: {unique_classes_full_raw}")
    if num_classes < 2:
        raise ValueError(f"Number of classes must be at least 2. Found {num_classes}.")

    # --- Preprocessing Setup ---
    # Define numerical and categorical columns from config - these must match columns in loaded data
    preproc_cfg = config.get('preprocessing', {})
    numerical_cols = preproc_cfg.get('numerical_cols', X_full_raw_df.select_dtypes(include=np.number).columns.tolist() if isinstance(X_full_raw_df, pd.DataFrame) else [])
    categorical_cols = preproc_cfg.get('categorical_cols', X_full_raw_df.select_dtypes(include='object').columns.tolist() if isinstance(X_full_raw_df, pd.DataFrame) else [])

    # Ensure numerical_cols and categorical_cols are disjoint and present in the DataFrame
    if isinstance(X_full_raw_df, pd.DataFrame):
        all_cols = X_full_raw_df.columns.tolist()
        numerical_cols = [col for col in numerical_cols if col in all_cols]
        categorical_cols = [col for col in categorical_cols if col in all_cols]
        # Handle potential overlap by removing from numerical if also in categorical (or vice-versa, based on policy)
        numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

    logger.info(f"Using numerical columns for preprocessing: {numerical_cols}")
    logger.info(f"Using categorical columns for preprocessing: {categorical_cols}")

    # Global preprocessor - will be fit on outer train folds
    # Note: For NCV, the preprocessor should be fit *inside each outer fold* on its training split.
    # This `global_preprocessor` is a template; a new one is instantiated per outer fold.

    # --- Nested Cross-Validation Setup ---
    n_outer_folds = config.get('ensemble', {}).get('n_outer_folds', 5)
    n_inner_folds = config.get('ensemble', {}).get('n_inner_folds_for_oof', 5)
    outer_skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)

    outer_fold_metrics_meta = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': []}
    outer_fold_metrics_soft_vote = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': []}

    # Convert to NumPy for SKFold if not already (assuming X_full_raw_df and y_full_raw_series are pandas)
    X_full_for_split = X_full_raw_df.to_numpy() if isinstance(X_full_raw_df, pd.DataFrame) else X_full_raw_df
    y_full_for_split = y_full_raw_series.to_numpy() if isinstance(y_full_raw_series, pd.Series) else y_full_raw_series


    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_skf.split(X_full_for_split, y_full_for_split)):
        logger.info(f"===== Starting Outer Fold {outer_fold_idx + 1}/{n_outer_folds} =====")

        # Get DataFrame/Series slices for this outer fold
        X_outer_train_raw_fold_df = X_full_raw_df.iloc[outer_train_idx]
        y_outer_train_fold_series = y_full_raw_series.iloc[outer_train_idx]
        X_outer_test_raw_fold_df = X_full_raw_df.iloc[outer_test_idx]
        y_outer_test_fold_series = y_full_raw_series.iloc[outer_test_idx]

        logger.debug(f"Outer Fold {outer_fold_idx+1}: X_outer_train_raw_fold_df shape {X_outer_train_raw_fold_df.shape}, X_outer_test_raw_fold_df shape {X_outer_test_raw_fold_df.shape}")

        # --- Preprocessing for the current outer fold ---
        # Fit preprocessor on this outer fold's training data
        fold_preprocessor = get_preprocessor(
            numerical_cols=numerical_cols, # Use list of names
            categorical_cols=categorical_cols, # Use list of names
            imputation_strategy=preproc_cfg.get('imputation_strategy', 'median'),
            scale_numerics=preproc_cfg.get('scale_numerics', True),
            handle_unknown_categorical=preproc_cfg.get('onehot_handle_unknown', 'ignore')
        )
        try:
            logger.info(f"Outer Fold {outer_fold_idx+1}: Fitting preprocessor on X_outer_train_raw_fold_df...")
            # fit_transform expects DataFrame
            X_outer_train_processed = fold_preprocessor.fit_transform(X_outer_train_raw_fold_df)
            X_outer_test_processed = fold_preprocessor.transform(X_outer_test_raw_fold_df)

            # Get feature names after transformation for TECO if needed
            try:
                processed_feature_names = fold_preprocessor.get_feature_names_out()
            except Exception: # Older sklearn might not have get_feature_names_out or it might fail
                # Fallback: generate generic names if needed, or ensure models can handle raw numpy
                num_processed_features = X_outer_train_processed.shape[1]
                processed_feature_names = [f'proc_feat_{i}' for i in range(num_processed_features)]
                logger.warning(f"Could not get feature names from preprocessor. Using generic names: {processed_feature_names[:5]}...")


            y_outer_train = y_outer_train_fold_series.to_numpy() # Labels to numpy
            y_outer_test = y_outer_test_fold_series.to_numpy()
            logger.info(f"Outer Fold {outer_fold_idx+1}: Preprocessing complete. X_outer_train_processed shape {X_outer_train_processed.shape}, X_outer_test_processed shape {X_outer_test_processed.shape}")
        except Exception as e:
            logger.error(f"Outer Fold {outer_fold_idx+1}: Error during general preprocessing: {e}. Using raw data for this outer fold (if numpy).")
            # Fallback to numpy versions if DataFrames were used
            X_outer_train_processed = X_outer_train_raw_fold_df.to_numpy()
            X_outer_test_processed = X_outer_test_raw_fold_df.to_numpy()
            y_outer_train = y_outer_train_fold_series.to_numpy()
            y_outer_test = y_outer_test_fold_series.to_numpy()
            processed_feature_names = X_outer_train_raw_fold_df.columns.tolist()


        # --- Inner Cross-Validation for OOF generation ---
        inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + outer_fold_idx)
        oof_preds_inner = {
            'lgbm': np.zeros((len(y_outer_train), num_classes)),
            'teco': np.zeros((len(y_outer_train), num_classes)),
            'stm_gnn': np.zeros((len(y_outer_train), num_classes))
        }
        base_model_preds_on_outer_test_sum = {
            'lgbm': np.zeros((len(y_outer_test), num_classes)),
            'teco': np.zeros((len(y_outer_test), num_classes)),
            'stm_gnn': np.zeros((len(y_outer_test), num_classes))
        }

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_skf.split(X_outer_train_processed, y_outer_train)):
            logger.info(f"--- Starting Inner Fold {inner_fold_idx + 1}/{n_inner_folds} (Outer Fold {outer_fold_idx+1}) ---")
            X_inner_fold_train, y_inner_fold_train = X_outer_train_processed[inner_train_idx], y_outer_train[inner_train_idx]
            X_inner_fold_val, y_inner_fold_val = X_outer_train_processed[inner_val_idx], y_outer_train[inner_val_idx]

            if config.get('balancing', {}).get('use_rsmote_gan_in_cv', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Applying RSMOTE-GAN...")
                rsmote_cv_config = config['balancing'].get('rsmote_gan_params', {})
                rsmote_gan_cv = RSMOTEGAN(
                    k_neighbors=rsmote_cv_config.get('k', 5),
                    minority_upsample_factor=rsmote_cv_config.get('minority_upsample_factor', 3.0),
                    random_state=seed + outer_fold_idx + inner_fold_idx
                )
                try:
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = rsmote_gan_cv.fit_resample(X_inner_fold_train, y_inner_fold_train)
                    logger.info(f"Inner Fold {inner_fold_idx+1}: RSMOTE-GAN completed. New shape: {X_inner_fold_train_balanced.shape}")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during RSMOTE-GAN: {e}. Proceeding without balancing.")
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = X_inner_fold_train, y_inner_fold_train
            else:
                X_inner_fold_train_balanced, y_inner_fold_train_balanced = X_inner_fold_train, y_inner_fold_train

            # --- 1. Train LightGBM (Inner Fold) ---
            if config.get('ensemble', {}).get('train_lgbm', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Training LightGBM...")
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
                        num_boost_round=lgbm_config.get('num_boost_round', 1000), # Full run params
                        early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 50), # Full run params
                        verbose=False
                    )
                    oof_preds_inner['lgbm'][inner_val_idx] = lgbm_inner_fold_model.predict_proba(X_inner_fold_val)
                    base_model_preds_on_outer_test_sum['lgbm'] += lgbm_inner_fold_model.predict_proba(X_outer_test_processed) / n_inner_folds
                    logger.info(f"Inner Fold {inner_fold_idx+1}: LightGBM training and prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during LightGBM: {e}")
                    # Fill with default (e.g. uniform) if error, to avoid breaking concatenation
                    oof_preds_inner['lgbm'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1/num_classes)
                    base_model_preds_on_outer_test_sum['lgbm'] += np.full((len(y_outer_test), num_classes), 1/num_classes) / n_inner_folds


            # --- 2. Train TECO-Transformer (Inner Fold) ---
            if config.get('ensemble', {}).get('train_teco', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Training TECO-Transformer...")
                try:
                    teco_config = config.get('ensemble', {}).get('teco_params', {})

                    # Use processed_feature_names from the outer fold's preprocessor
                    # These names correspond to the columns in X_inner_fold_train_balanced etc.
                    # Ensure the feature names are consistent with what TabularSequenceDataset expects
                    df_inner_fold_train_teco = pd.DataFrame(X_inner_fold_train_balanced, columns=processed_feature_names)
                    df_inner_fold_val_teco = pd.DataFrame(X_inner_fold_val, columns=processed_feature_names)
                    df_outer_test_teco = pd.DataFrame(X_outer_test_processed, columns=processed_feature_names)

                    teco_target_column_name = 'target_for_teco' # Internal name for TabularSequenceDataset

                    train_teco_dataset_inner = TabularSequenceDataset(
                        data_frame=df_inner_fold_train_teco,
                        targets=y_inner_fold_train_balanced,
                        feature_columns=processed_feature_names, # Pass all processed feature names
                        target_column_name=teco_target_column_name
                    )
                    val_teco_dataset_inner = TabularSequenceDataset(df_inner_fold_val_teco, y_inner_fold_val, processed_feature_names, teco_target_column_name)
                    outer_test_teco_dataset = TabularSequenceDataset(df_outer_test_teco, np.zeros(len(df_outer_test_teco)), processed_feature_names, teco_target_column_name)

                    batch_size_teco = teco_config.get('batch_size_teco', 32)
                    train_teco_loader_inner = DataLoader(train_teco_dataset_inner, batch_size=batch_size_teco, shuffle=True, collate_fn=basic_collate_fn)
                    val_teco_loader_inner = DataLoader(val_teco_dataset_inner, batch_size=batch_size_teco, shuffle=False, collate_fn=basic_collate_fn)
                    outer_test_teco_loader = DataLoader(outer_test_teco_dataset, batch_size=batch_size_teco, shuffle=False, collate_fn=basic_collate_fn)

                    teco_model_inner = TECOTransformerModel(
                        input_feature_dim=len(processed_feature_names), # From preprocessed data
                        d_model=teco_config.get('d_model', 512),
                        num_encoder_layers=teco_config.get('num_encoder_layers', 4),
                        nhead=teco_config.get('nhead', 8),
                        dim_feedforward=teco_config.get('dim_feedforward', 2048),
                        dropout=teco_config.get('dropout', 0.1),
                        num_classes=num_classes,
                        max_seq_len=teco_config.get('max_seq_len', 100) # Adjust if sequences are longer
                    ).to(device)

                    teco_criterion_inner = nn.CrossEntropyLoss() # Add class weights if needed from config
                    teco_optimizer_inner = optim.Adam(teco_model_inner.parameters(), lr=teco_config.get('lr_teco', 1e-4))
                    epochs_teco_inner = teco_config.get('epochs_teco_inner', 10) # Full run epochs

                    for epoch in range(epochs_teco_inner):
                        teco_model_inner.train()
                        epoch_loss_sum = 0.0
                        for batch in train_teco_loader_inner:
                            teco_optimizer_inner.zero_grad()
                            # Ensure batch items are on the correct device
                            sequences = batch['sequence'].to(device)
                            padding_masks = batch['padding_mask'].to(device)
                            targets = batch['target'].to(device)

                            outputs = teco_model_inner(sequences, padding_masks)
                            loss = teco_criterion_inner(outputs, targets)
                            loss.backward()
                            teco_optimizer_inner.step()
                            epoch_loss_sum += loss.item()
                        logger.debug(f"Inner Fold {inner_fold_idx+1}, TECO Epoch {epoch+1}/{epochs_teco_inner}, Avg Train Loss: {epoch_loss_sum/len(train_teco_loader_inner):.4f}")

                    teco_model_inner.eval()
                    inner_val_preds_teco_list = []
                    with torch.no_grad():
                        for batch in val_teco_loader_inner:
                            outputs = teco_model_inner(batch['sequence'].to(device), batch['padding_mask'].to(device))
                            inner_val_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

                    if inner_val_preds_teco_list: # Check if list is not empty
                        oof_preds_inner['teco'][inner_val_idx] = np.concatenate(inner_val_preds_teco_list, axis=0)[:, :num_classes]
                    else: # Handle case with no validation samples or error
                         oof_preds_inner['teco'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1/num_classes)


                    outer_test_preds_teco_list = []
                    with torch.no_grad():
                        for batch in outer_test_teco_loader:
                            outputs = teco_model_inner(batch['sequence'].to(device), batch['padding_mask'].to(device))
                            outer_test_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())

                    if outer_test_preds_teco_list:
                        base_model_preds_on_outer_test_sum['teco'] += np.concatenate(outer_test_preds_teco_list, axis=0)[:, :num_classes] / n_inner_folds
                    else: # Handle empty list
                        base_model_preds_on_outer_test_sum['teco'] += np.full((len(y_outer_test), num_classes), 1/num_classes) / n_inner_folds

                    logger.info(f"Inner Fold {inner_fold_idx+1}: TECO-Transformer training and prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during TECO-Transformer: {e}")
                    oof_preds_inner['teco'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1/num_classes)
                    base_model_preds_on_outer_test_sum['teco'] += np.full((len(y_outer_test), num_classes), 1/num_classes) / n_inner_folds


            # --- 3. Train STM-GNN (Inner Fold - Conceptual: Needs real data loader and graph features) ---
            if config.get('ensemble', {}).get('train_stm_gnn', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Training STM-GNN (Conceptual - requires graph data)...")
                # STM-GNN requires graph-structured data (node features, edge indices per snapshot)
                # This part needs a dedicated graph data loader and feature engineering pipeline.
                # For now, we'll use placeholder predictions as in the original script.
                try:
                    stm_gnn_config = config.get('ensemble',{}).get('stm_gnn_params',{})
                    # This is a placeholder. Real STM-GNN training is complex.
                    # 1. Prepare graph data for STM-GNN (e.g., from X_inner_fold_train_balanced)
                    #    This would involve creating graph snapshots.
                    #    num_node_features_stm = X_inner_fold_train_balanced.shape[1] # if using tabular as node features

                    # For this conceptual run, simulate predictions based on class distribution
                    class_probs_stm = np.bincount(y_inner_fold_train_balanced) / len(y_inner_fold_train_balanced) if len(y_inner_fold_train_balanced) > 0 else np.full(num_classes, 1/num_classes)
                    if len(class_probs_stm) < num_classes: # Ensure correct shape if some classes are missing
                        temp_p = np.full(num_classes, 1e-6) # Small probability for missing classes
                        temp_p[:len(class_probs_stm)] = class_probs_stm
                        class_probs_stm = temp_p / np.sum(temp_p)

                    num_val_samples_stm = len(y_inner_fold_val)
                    dummy_stm_oof = np.random.multinomial(1, class_probs_stm, size=num_val_samples_stm) * 0.8 + \
                                    np.random.rand(num_val_samples_stm, num_classes) * 0.2
                    oof_preds_inner['stm_gnn'][inner_val_idx] = dummy_stm_oof / np.sum(dummy_stm_oof, axis=1, keepdims=True)

                    num_outer_test_samples_stm = len(y_outer_test)
                    dummy_stm_test = np.random.multinomial(1, class_probs_stm, size=num_outer_test_samples_stm) * 0.8 + \
                                     np.random.rand(num_outer_test_samples_stm, num_classes) * 0.2
                    base_model_preds_on_outer_test_sum['stm_gnn'] += (dummy_stm_test / np.sum(dummy_stm_test, axis=1, keepdims=True)) / n_inner_folds
                    logger.info(f"Inner Fold {inner_fold_idx+1}: STM-GNN (conceptual placeholder) prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during STM-GNN (conceptual): {e}")
                    oof_preds_inner['stm_gnn'][inner_val_idx] = np.full((len(inner_val_idx), num_classes), 1/num_classes)
                    base_model_preds_on_outer_test_sum['stm_gnn'] += np.full((len(y_outer_test), num_classes), 1/num_classes) / n_inner_folds

        # --- Meta-Learner Training and Evaluation for Outer Fold ---
        logger.info(f"Outer Fold {outer_fold_idx+1}: Finished generating OOF predictions from inner CV.")
        meta_features_train_outer_list = []
        if config.get('ensemble', {}).get('train_lgbm', True): meta_features_train_outer_list.append(oof_preds_inner['lgbm'])
        if config.get('ensemble', {}).get('train_teco', True): meta_features_train_outer_list.append(oof_preds_inner['teco'])
        if config.get('ensemble', {}).get('train_stm_gnn', True): meta_features_train_outer_list.append(oof_preds_inner['stm_gnn'])

        if not meta_features_train_outer_list:
            logger.error(f"Outer Fold {outer_fold_idx+1}: No base models for meta-learner. Skipping.")
        else:
            X_meta_train_outer = np.concatenate(meta_features_train_outer_list, axis=1)
            y_meta_train_outer = y_outer_train
            logger.info(f"Outer Fold {outer_fold_idx+1}: Meta-learner training features shape: {X_meta_train_outer.shape}")

            if config.get('ensemble', {}).get('train_meta_learner', True):
                logger.info(f"Outer Fold {outer_fold_idx+1}: Training XGBoost Meta-Learner...")
                try:
                    meta_config = config.get('ensemble', {}).get('meta_learner_xgb_params', {})
                    xgb_meta_model_outer = XGBoostMetaLearner(
                        params=meta_config.get('model_specific_params'),
                        depth=meta_config.get('depth', 3)
                    )
                    xgb_meta_model_outer.train(
                        X_meta_train_outer, y_meta_train_outer,
                        num_boost_round=meta_config.get('num_boost_round', 200), # Full run
                        early_stopping_rounds=meta_config.get('early_stopping_rounds', 20) # Full run
                    )
                    logger.info(f"Outer Fold {outer_fold_idx+1}: XGBoost Meta-Learner trained.")

                    meta_features_test_outer_list = []
                    if config.get('ensemble', {}).get('train_lgbm', True): meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['lgbm'])
                    if config.get('ensemble', {}).get('train_teco', True): meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['teco'])
                    if config.get('ensemble', {}).get('train_stm_gnn', True): meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['stm_gnn'])

                    X_meta_test_outer = np.concatenate(meta_features_test_outer_list, axis=1)
                    final_preds_meta_proba_outer = xgb_meta_model_outer.predict_proba(X_meta_test_outer)
                    final_preds_meta_labels_outer = xgb_meta_model_outer.predict(X_meta_test_outer)

                    acc_meta_outer = accuracy_score(y_outer_test, final_preds_meta_labels_outer)
                    f1_meta_outer = f1_score(y_outer_test, final_preds_meta_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    prec_meta_outer = precision_score(y_outer_test, final_preds_meta_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    rec_meta_outer = recall_score(y_outer_test, final_preds_meta_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    auroc_meta_outer = -1.0
                    try:
                        probas_for_auc = final_preds_meta_proba_outer[:, 1] if num_classes == 2 and final_preds_meta_proba_outer.ndim == 2 and final_preds_meta_proba_outer.shape[1] >=2 else final_preds_meta_proba_outer
                        auroc_meta_outer = roc_auc_score(y_outer_test, probas_for_auc, multi_class='ovr', average='weighted')
                    except ValueError as e:
                        logger.warning(f"Outer Fold {outer_fold_idx+1} Meta AUROC calc error: {e}. Proba shape: {final_preds_meta_proba_outer.shape}")

                    outer_fold_metrics_meta['accuracy'].append(acc_meta_outer)
                    outer_fold_metrics_meta['auroc'].append(auroc_meta_outer) # Key metric for GTscore
                    outer_fold_metrics_meta['f1'].append(f1_meta_outer)
                    outer_fold_metrics_meta['precision'].append(prec_meta_outer)
                    outer_fold_metrics_meta['recall'].append(rec_meta_outer)
                    wandb.log({f"outer_fold_{outer_fold_idx+1}/meta_auroc": auroc_meta_outer, "outer_fold": outer_fold_idx + 1})
                    logger.info(f"Outer Fold {outer_fold_idx+1} Meta-Learner: AUROC={auroc_meta_outer:.4f}, Acc={acc_meta_outer:.4f}")
                except Exception as e:
                    logger.error(f"Outer Fold {outer_fold_idx+1}: Error during Meta-Learner: {e}")
                    for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)

        # --- Soft Voting Evaluation for Outer Fold ---
        soft_vote_weights = config.get('ensemble', {}).get('soft_vote_weights', {})
        if soft_vote_weights and any(config.get('ensemble', {}).get(f'train_{model_key}', False) for model_key in soft_vote_weights):
            logger.info(f"Outer Fold {outer_fold_idx+1}: Performing Soft Voting...")
            # ... (Soft voting logic remains largely the same, ensure it uses base_model_preds_on_outer_test_sum correctly) ...
            # Ensure this part is also robust to model failures and logs AUROC
            try:
                final_preds_soft_vote_proba_outer = np.zeros((len(y_outer_test), num_classes))
                total_weight = 0.0 # Ensure float for division
                active_models_count = 0

                if config.get('ensemble', {}).get('train_lgbm', True) and 'lgbm' in soft_vote_weights:
                    weight = soft_vote_weights['lgbm']
                    final_preds_soft_vote_proba_outer += weight * base_model_preds_on_outer_test_sum['lgbm']
                    total_weight += weight
                    active_models_count +=1
                if config.get('ensemble', {}).get('train_teco', True) and 'teco' in soft_vote_weights:
                    weight = soft_vote_weights['teco']
                    final_preds_soft_vote_proba_outer += weight * base_model_preds_on_outer_test_sum['teco']
                    total_weight += weight
                    active_models_count +=1
                if config.get('ensemble', {}).get('train_stm_gnn', True) and 'stm_gnn' in soft_vote_weights:
                    weight = soft_vote_weights['stm_gnn']
                    final_preds_soft_vote_proba_outer += weight * base_model_preds_on_outer_test_sum['stm_gnn']
                    total_weight += weight
                    active_models_count +=1

                if active_models_count > 0 and total_weight > 1e-6 : # Avoid division by zero or tiny weights
                    # Normalize if weights don't sum to 1 (or if only some models contributed)
                    # final_preds_soft_vote_proba_outer /= total_weight # if weights are just ratios
                    # Ensure probabilities sum to 1 per sample after weighting and summing
                    row_sums = final_preds_soft_vote_proba_outer.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1 # Avoid division by zero
                    final_preds_soft_vote_proba_outer = final_preds_soft_vote_proba_outer / row_sums

                    final_preds_soft_vote_labels_outer = np.argmax(final_preds_soft_vote_proba_outer, axis=1)
                    acc_sv_outer = accuracy_score(y_outer_test, final_preds_soft_vote_labels_outer)
                    f1_sv_outer = f1_score(y_outer_test, final_preds_soft_vote_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    auroc_sv_outer = -1.0
                    try:
                        probas_for_auc_sv = final_preds_soft_vote_proba_outer[:, 1] if num_classes == 2 and final_preds_soft_vote_proba_outer.ndim == 2 and final_preds_soft_vote_proba_outer.shape[1] >=2 else final_preds_soft_vote_proba_outer
                        auroc_sv_outer = roc_auc_score(y_outer_test, probas_for_auc_sv, multi_class='ovr', average='weighted')
                    except ValueError as e:
                        logger.warning(f"Outer Fold {outer_fold_idx+1} SoftVote AUROC calculation error: {e}. Proba shape: {final_preds_soft_vote_proba_outer.shape}")

                    outer_fold_metrics_soft_vote['auroc'].append(auroc_sv_outer) # Key metric
                    outer_fold_metrics_soft_vote['accuracy'].append(acc_sv_outer)
                    # ... other metrics ...
                    wandb.log({f"outer_fold_{outer_fold_idx+1}/sv_auroc": auroc_sv_outer, "outer_fold": outer_fold_idx + 1})
                    logger.info(f"Outer Fold {outer_fold_idx+1} Soft Vote: AUROC={auroc_sv_outer:.4f}, Acc={acc_sv_outer:.4f}")
                else:
                    logger.warning(f"Outer Fold {outer_fold_idx+1}: Soft Voting not performed (no active models or zero total weight).")
                    for key in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key].append(np.nan)

            except Exception as e:
                logger.error(f"Outer Fold {outer_fold_idx+1}: Error during Soft Voting: {e}")
                for key in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key].append(np.nan)


    # --- Nested Cross-Validation Summary ---
    logger.info("===== Nested Cross-Validation Summary =====")
    # Log average AUROC for meta-learner (primary GTscore optimization target)
    if config.get('ensemble', {}).get('train_meta_learner', True) and len(outer_fold_metrics_meta['auroc']) > 0:
        avg_meta_auroc = np.nanmean(outer_fold_metrics_meta['auroc'])
        std_meta_auroc = np.nanstd(outer_fold_metrics_meta['auroc'])
        logger.info(f"Meta-Learner Average AUROC: {avg_meta_auroc:.4f} +/- {std_meta_auroc:.4f}")
        wandb.summary["ncv_meta_avg_auroc"] = avg_meta_auroc # Key summary metric
        wandb.summary["ncv_meta_std_auroc"] = std_meta_auroc
        # Log other average metrics as well
        for metric_name, values in outer_fold_metrics_meta.items():
            if metric_name not in ['auroc']: # AUROC already logged with std
                 avg_val = np.nanmean(values)
                 wandb.summary[f"ncv_meta_avg_{metric_name}"] = avg_val
                 logger.info(f"Meta-Learner Average {metric_name.capitalize()}: {avg_val:.4f}")


    if soft_vote_weights and sum(soft_vote_weights.values()) > 0 and len(outer_fold_metrics_soft_vote['auroc']) > 0:
        avg_sv_auroc = np.nanmean(outer_fold_metrics_soft_vote['auroc'])
        std_sv_auroc = np.nanstd(outer_fold_metrics_soft_vote['auroc'])
        logger.info(f"Soft Voting Average AUROC: {avg_sv_auroc:.4f} +/- {std_sv_auroc:.4f}")
        wandb.summary["ncv_sv_avg_auroc"] = avg_sv_auroc
        wandb.summary["ncv_sv_std_auroc"] = std_sv_auroc
        # Log other average metrics for soft voting
        for metric_name, values in outer_fold_metrics_soft_vote.items():
            if metric_name not in ['auroc']:
                 avg_val = np.nanmean(values)
                 wandb.summary[f"ncv_sv_avg_{metric_name}"] = avg_val
                 logger.info(f"Soft Voting Average {metric_name.capitalize()}: {avg_val:.4f}")

    wandb.finish()
    logger.info("Full Nested Cross-Validation ensemble training run finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main training script for Clinical Prediction Model with NCV.")
    parser.add_argument('--config', type=str, default='configs/dummy_train_config.yaml', # Or point to a new 'full_train_config.yaml'
                        help='Path to the training configuration file.')
    args = parser.parse_args()

    # The part that auto-creates dummy_train_config.yaml can be removed or adapted
    # if we expect a dedicated config file for full runs.
    # For now, let's assume the dummy_train_config.yaml will be updated for "fuller" settings.
    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} not found. Please create it.")
        # Optionally, create a more comprehensive default config here if needed for a "full run"
        # For now, it relies on the existing dummy config creation logic if file is missing,
        # which might not be ideal for a "full" run.
        # Consider removing auto-creation for full runs and requiring a specific config.
        # Fallback to original dummy creation for now if not found
        print(f"Warning: Configuration file {args.config} not found. Attempting to create a dummy one.")
        # ... (dummy config creation code from original script - may need updates for full run defaults)
        # This dummy creation should ideally be in a separate utility or test setup.
        # For this task, we assume the user will provide an appropriate config.
        sys.exit(1) # Exit if config is not found, rather than creating a minimal dummy for a full run.


    main(args.config)
