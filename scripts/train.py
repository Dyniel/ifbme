import sys
import os
# Add the project root to sys.path
# This assumes scripts/train.py is two levels down from the project root (e.g., project_root/scripts/train.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml # For loading configurations
import wandb
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging # For structured logging

# Project-specific imports
from data_utils.balancing import RSMOTEGAN
# from data_utils.losses import ClassBalancedFocalLoss # Not directly used at top level after NCV refactor
from models import LightGBMModel, XGBoostMetaLearner
from models.teco_transformer import TECOTransformerModel
from data_utils.sequence_loader import TabularSequenceDataset, basic_collate_fn
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# --- Logger Setup ---
# Configure logger for the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    stream=sys.stdout # Log to console
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Loads YAML configuration file.
    Includes basic error handling for file operations.
    """
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

def get_samples_per_class(y_train): # This helper might still be used if ClassBalancedFocalLoss is used for TECO
    """Helper function to count samples per class from training labels."""
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    unique, counts = np.unique(y_train, return_counts=True)
    # Ensure counts are in the order of class indices if they are not contiguous
    # For now, assume unique contains class indices sorted, e.g., [0, 1, 2]
    # A more robust way would be to create a zero array for num_classes and fill it.
    # samples = np.zeros(N_CLASSES)
    # for cls_idx, count in zip(unique, counts): samples[cls_idx] = count
    return counts.tolist()


def main(config_path):
    """Main training loop."""
    config = load_config(config_path)

    # --- Initialize W&B ---
    # Potentially load wandb specific configs from main config file
    wandb_config = config.get('wandb', {})
    wandb.init(
        project=wandb_config.get('project', 'ifbme-project'), # Replace with your project name
        entity=wandb_config.get('entity', None), # Replace with your entity (user or team)
        config=config, # Log the entire configuration
        name=wandb_config.get('run_name', None), # Optional: set a run name
        notes=wandb_config.get('run_notes', None), # Optional: add notes
        tags=wandb_config.get('tags', None) # Optional: add tags
    )
    # Update wandb config with the loaded config (in case init did not take it all)
    # wandb.config.update(config) # Already done by passing config to wandb.init
    logger.info(f"W&B initialized for project '{wandb_config.get('project', 'ifbme-project')}'")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu")
    logger.info(f"Using device: {device}")

    # Seed for reproducibility
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    # --- Data Loading and Preprocessing ---
    logger.info("Starting data loading and preprocessing for NCV...")

    # --- Conceptual Data Loading using data_utils.data_loader ---
    # In a real scenario, X_full_raw and y_full_raw would be loaded here.
    # from data_utils.data_loader import load_raw_data
    # try:
    #     # Assuming config has 'data_paths' and 'target_column' correctly set up
    #     # X_full_raw_df, y_full_raw_series = load_raw_data(config) # If load_raw_data returns df/series
    #     # X_full_raw = X_full_raw_df.to_numpy() # Convert to numpy as expected by current code
    #     # y_full_raw = y_full_raw_series.to_numpy()
    #     logger.info("Conceptual: Real data loading would occur here.")
    # except Exception as e:
    #     logger.critical(f"Failed to load raw data: {e}. Exiting.")
    #     sys.exit(1)
    # For this conceptual step, we continue with dummy data generation if real loading fails or is bypassed.
    logger.info("Using dummy data generation for this run.")
    num_samples_train_orig = config.get('dummy_data_train_samples', 400) # Adjusted for dummy config
    num_samples_val_orig = config.get('dummy_data_val_samples', 100)  # Adjusted for dummy config
    num_total_samples = num_samples_train_orig + num_samples_val_orig
    num_features = config.get('dummy_data_features', 20)

    # Determine num_classes from config first to ensure consistency
    # This was previously derived from y_train_raw later, which is fine, but for NCV setup,
    # it's good to have it early.
    num_classes_config = config.get('dummy_data_classes', 2)
    if num_classes_config < 2:
        raise ValueError(f"dummy_data_classes in config must be at least 2. Found {num_classes_config}.")

    # Create imbalanced dummy data for the combined dataset
    # Use weights from original train config for simplicity, or define new combined weights
    weights_combined = config.get('dummy_data_train_weights', [0.9, 0.1]) # Or new 'dummy_data_combined_weights'
    p_combined = np.array(weights_combined) / np.sum(weights_combined)

    # Generate features for the entire dataset
    X_full_raw = np.random.rand(num_total_samples, num_features)
    # Generate labels for the entire dataset
    y_full_raw = np.random.choice(num_classes_config, num_total_samples, p=p_combined)

    logger.info(f"Full raw data shape for NCV: X={X_full_raw.shape}, y={y_full_raw.shape}")
    logger.info(f"Full raw data class distribution: {np.bincount(y_full_raw)}")

    # The initial top-level balancing (e.g., config.get('balancing', {}).get('use_rsmote_gan', False))
    # is removed from this top level. Balancing, if used, should primarily happen:
    # 1. Inside the inner CV loop on its specific training folds (controlled by 'use_rsmote_gan_in_cv').
    # 2. Optionally, on the outer loop's training portion (X_outer_train) before the inner CV loop begins.
    #    This would require a new configuration flag, e.g., 'use_rsmote_gan_on_outer_train'.
    #    For now, we rely on 'use_rsmote_gan_in_cv' for balancing within the inner folds.

    # The direct conversion to torch tensors and model definition/loss/optimizer for a single model
    # (previously done here) is no longer relevant at this top level, as all model training
    # now occurs inside the Nested Cross-Validation structure.

    logger.info("Starting Nested Cross-Validation ensemble training process...")

    # --- Nested Cross-Validation Setup ---
    # Number of folds for the outer loop (evaluating the entire pipeline)
    n_outer_folds = config.get('ensemble', {}).get('n_outer_folds', 5)
    # Number of folds for the inner loop (generating OOF predictions for meta-learner training)
    n_inner_folds = config.get('ensemble', {}).get('n_inner_folds_for_oof', 5) # Renamed from n_folds_for_oof

    outer_skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)

    # Store metrics from each outer fold
    outer_fold_metrics_meta = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': []}
    outer_fold_metrics_soft_vote = {'accuracy': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': []}

    # Determine num_classes from the full dataset's labels
    # This is crucial for initializing OOF arrays and metrics correctly throughout the NCV process.
    unique_classes_full_raw = np.unique(y_full_raw)
    num_classes = len(unique_classes_full_raw) # This 'num_classes' will be used globally within main()
    logger.info(f"Determined number of classes for NCV: {num_classes} from y_full_raw unique values: {unique_classes_full_raw}")
    if num_classes < 2:
        logger.error(f"Number of classes must be at least 2 for classification. Found {num_classes} in y_full_raw for NCV.")
        raise ValueError(f"Number of classes must be at least 2. Found {num_classes} in y_full_raw for NCV.")


    # --- Outer Cross-Validation Loop ---
    # This loop iterates through the outer folds. In each iteration:
    # - The data is split into an outer training set (X_outer_train, y_outer_train) and an outer test set (X_outer_test, y_outer_test).
    # - The inner CV loop runs on X_outer_train to train base models and generate OOFs for the meta-learner.
    # - The meta-learner is trained on these OOFs.
    # - The entire ensemble (base models + meta-learner) is then evaluated on X_outer_test.
    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_skf.split(X_full_raw, y_full_raw)):
        logger.info(f"===== Starting Outer Fold {outer_fold_idx + 1}/{n_outer_folds} =====")
        X_outer_train_raw_fold, y_outer_train_fold = X_full_raw[outer_train_idx], y_full_raw[outer_train_idx]
        X_outer_test_raw_fold, y_outer_test_fold = X_full_raw[outer_test_idx], y_full_raw[outer_test_idx]
        logger.debug(f"Outer Fold {outer_fold_idx+1}: X_outer_train_raw_fold shape {X_outer_train_raw_fold.shape}, X_outer_test_raw_fold shape {X_outer_test_raw_fold.shape}")

        # --- Conceptual General Preprocessing ---
        # from data_utils.preprocess import get_preprocessor
        # numerical_cols = config.get('preprocessing',{}).get('numerical_cols', []) # Define in config
        # categorical_cols = config.get('preprocessing',{}).get('categorical_cols', []) # Define in config
        # preproc_config = config.get('preprocessing', {})
        #
        # preprocessor = get_preprocessor(
        #     numerical_cols=numerical_cols, categorical_cols=categorical_cols,
        #     imputation_strategy=preproc_config.get('imputation_strategy', 'median'),
        #     scale_numerics=preproc_config.get('scale_numerics', True),
        #     handle_unknown_categorical=preproc_config.get('onehot_handle_unknown', 'ignore')
        # )
        #
        # try:
        #     logger.info(f"Outer Fold {outer_fold_idx+1}: Fitting preprocessor on X_outer_train_raw_fold...")
        #     X_outer_train_processed = preprocessor.fit_transform(pd.DataFrame(X_outer_train_raw_fold, columns=[f'feature_{i}' for i in range(X_outer_train_raw_fold.shape[1])])) # Assuming feature names for DataFrame conversion
        #     X_outer_test_processed = preprocessor.transform(pd.DataFrame(X_outer_test_raw_fold, columns=[f'feature_{i}' for i in range(X_outer_test_raw_fold.shape[1])]))
        #     y_outer_train = y_outer_train_fold # Labels usually don't need this kind of preprocessing
        #     y_outer_test = y_outer_test_fold
        #     logger.info(f"Outer Fold {outer_fold_idx+1}: Preprocessing complete. X_outer_train_processed shape {X_outer_train_processed.shape}, X_outer_test_processed shape {X_outer_test_processed.shape}")
        # except Exception as e:
        #     logger.error(f"Outer Fold {outer_fold_idx+1}: Error during general preprocessing: {e}. Using raw data for this outer fold.")
        #     X_outer_train_processed = X_outer_train_raw_fold
        #     X_outer_test_processed = X_outer_test_raw_fold
        #     y_outer_train = y_outer_train_fold
        #     y_outer_test = y_outer_test_fold
        # For dummy data and current conceptual models, we'll bypass actual preprocessing and use raw splits.
        X_outer_train, y_outer_train = X_outer_train_raw_fold, y_outer_train_fold
        X_outer_test, y_outer_test = X_outer_test_raw_fold, y_outer_test_fold
        # --- End Conceptual General Preprocessing ---


        # --- Inner Cross-Validation for OOF generation on X_outer_train ---
        # This loop trains base models on folds of X_outer_train (processed) and collects their OOF predictions.
        # It also averages predictions from these base models on X_outer_test (processed).
        inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + outer_fold_idx) # Vary seed for robustness

        # Placeholders for OOF predictions from base models on X_outer_train
        oof_preds_inner = {
            'lgbm': np.zeros((len(y_outer_train), num_classes)),
            'teco': np.zeros((len(y_outer_train), num_classes)),
            'stm_gnn': np.zeros((len(y_outer_train), num_classes))
        }

        # Placeholders for base model predictions on X_outer_test (averaged over inner folds)
        base_model_preds_on_outer_test_sum = {
            'lgbm': np.zeros((len(y_outer_test), num_classes)),
            'teco': np.zeros((len(y_outer_test), num_classes)),
            'stm_gnn': np.zeros((len(y_outer_test), num_classes))
        }

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_skf.split(X_outer_train, y_outer_train)):
            logger.info(f"--- Starting Inner Fold {inner_fold_idx + 1}/{n_inner_folds} (Outer Fold {outer_fold_idx+1}) ---")
            X_inner_fold_train, y_inner_fold_train = X_outer_train[inner_train_idx], y_outer_train[inner_train_idx]
            X_inner_fold_val, y_inner_fold_val = X_outer_train[inner_val_idx], y_outer_train[inner_val_idx]
            logger.debug(f"Inner Fold {inner_fold_idx+1}: X_inner_fold_train shape {X_inner_fold_train.shape}, X_inner_fold_val shape {X_inner_fold_val.shape}")

            # Apply RSMOTE-GAN to (X_inner_fold_train, y_inner_fold_train) if configured
            # This balancing step is applied only to the training part of the current inner fold.
            if config.get('balancing', {}).get('use_rsmote_gan_in_cv', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Applying RSMOTE-GAN to inner fold training data...")
                rsmote_cv_config = config['balancing'].get('rsmote_gan_params', {})
                rsmote_gan_cv = RSMOTEGAN(
                    k_neighbors=rsmote_cv_config.get('k', 5),
                    minority_upsample_factor=rsmote_cv_config.get('minority_upsample_factor', 3.0),
                    random_state=seed + outer_fold_idx + inner_fold_idx # Vary seed
                )
                try:
                    X_inner_fold_train_balanced, y_inner_fold_train_balanced = rsmote_gan_cv.fit_resample(X_inner_fold_train, y_inner_fold_train)
                    logger.info(f"Inner Fold {inner_fold_idx+1}: RSMOTE-GAN completed. New shape: {X_inner_fold_train_balanced.shape}")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during RSMOTE-GAN: {e}. Proceeding without balancing for this fold.")
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
                    X_inner_fold_val, y_inner_fold_val, # Use inner val for early stopping
                    num_boost_round=lgbm_config.get('num_boost_round', 1000),
                    early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 50),
                    verbose=False # Reduce verbosity for inner loops
                )
                oof_preds_inner['lgbm'][inner_val_idx] = lgbm_inner_fold_model.predict_proba(X_inner_fold_val)
                base_model_preds_on_outer_test_sum['lgbm'] += lgbm_inner_fold_model.predict_proba(X_outer_test) / n_inner_folds
                logger.info(f"Inner Fold {inner_fold_idx+1}: LightGBM training and prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during LightGBM training or prediction: {e}")
                    # Ensure downstream processes can handle missing predictions if necessary,
                    # or fill with a default (e.g., zeros, though this might skew results).
                    # For now, if a model fails, its OOF/test predictions might remain zero.
                    pass # Continue to next model/fold

            # --- 2. Train TECO-Transformer (Inner Fold - Conceptual) ---
            if config.get('ensemble', {}).get('train_teco', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Training TECO-Transformer...")
                try:
                teco_config = config.get('ensemble', {}).get('teco_params', {})
                num_inner_fold_features = X_inner_fold_train_balanced.shape[1]
                teco_feature_columns = teco_config.get('feature_columns_teco', [f'feature_{i}' for i in range(num_inner_fold_features)])
                if len(teco_feature_columns) != num_inner_fold_features: # Fallback
                    teco_feature_columns = [f'feature_{i}' for i in range(num_inner_fold_features)]

                df_inner_fold_train = pd.DataFrame(X_inner_fold_train_balanced, columns=teco_feature_columns)
                df_inner_fold_val = pd.DataFrame(X_inner_fold_val, columns=teco_feature_columns)
                df_outer_test_teco = pd.DataFrame(X_outer_test, columns=teco_feature_columns)
                teco_target_column = teco_config.get('target_column_teco', 'outcomeType_teco')

                train_teco_dataset_inner = TabularSequenceDataset(None, df_inner_fold_train, y_inner_fold_train_balanced, teco_feature_columns, teco_target_column)
                val_teco_dataset_inner = TabularSequenceDataset(None, df_inner_fold_val, y_inner_fold_val, teco_feature_columns, teco_target_column)
                outer_test_teco_dataset = TabularSequenceDataset(None, df_outer_test_teco, np.zeros(len(df_outer_test_teco)), teco_feature_columns, teco_target_column) # Dummy y

                batch_size_teco = teco_config.get('batch_size_teco', 32)
                train_teco_loader_inner = DataLoader(train_teco_dataset_inner, batch_size_teco, shuffle=True, collate_fn=basic_collate_fn)
                val_teco_loader_inner = DataLoader(val_teco_dataset_inner, batch_size_batch_size_teco, shuffle=False, collate_fn=basic_collate_fn)
                outer_test_teco_loader = DataLoader(outer_test_teco_dataset, batch_size_teco, shuffle=False, collate_fn=basic_collate_fn)

                teco_model_inner = TECOTransformerModel(
                    input_feature_dim=len(teco_feature_columns),
                    d_model=teco_config.get('d_model', 64), # Smaller for faster NCV
                    num_encoder_layers=teco_config.get('num_encoder_layers', 1),
                    nhead=teco_config.get('nhead', 2),
                    dim_feedforward=teco_config.get('dim_feedforward', 128),
                    dropout=teco_config.get('dropout', 0.1),
                    num_classes=num_classes,
                    max_seq_len=teco_config.get('max_seq_len', 50)
                ).to(device)
                teco_criterion_inner = nn.CrossEntropyLoss()
                teco_optimizer_inner = optim.Adam(teco_model_inner.parameters(), lr=teco_config.get('lr_teco', 1e-4))
                epochs_teco_inner = teco_config.get('epochs_teco_inner', 3) # Fewer epochs for inner loop

                for _ in range(epochs_teco_inner): # Simplified training loop for brevity
                    teco_model_inner.train()
                    for batch in train_teco_loader_inner:
                        teco_optimizer_inner.zero_grad()
                        loss = teco_criterion_inner(teco_model_inner(batch['sequence'].to(device), batch['padding_mask'].to(device)), batch['target'].to(device))
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
                oof_preds_inner['teco'][inner_val_idx] = np.concatenate(inner_val_preds_teco_list, axis=0)[:, :num_classes]

                outer_test_preds_teco_list = []
                with torch.no_grad():
                    for batch in outer_test_teco_loader:
                        outputs = teco_model_inner(batch['sequence'].to(device), batch['padding_mask'].to(device))
                        outer_test_preds_teco_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
                base_model_preds_on_outer_test_sum['teco'] += np.concatenate(outer_test_preds_teco_list, axis=0)[:, :num_classes] / n_inner_folds
                logger.info(f"Inner Fold {inner_fold_idx+1}: TECO-Transformer training and prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during TECO-Transformer training or prediction: {e}")
                    pass # Continue

            # --- 3. Train STM-GNN (Inner Fold - Conceptual Placeholder) ---
            if config.get('ensemble', {}).get('train_stm_gnn', True):
                logger.info(f"Inner Fold {inner_fold_idx+1}: Training STM-GNN (Conceptual)...")
                try:
                    # Simplified dummy predictions for STM-GNN
                    class_probs_sim_inner = np.bincount(y_inner_fold_train) / len(y_inner_fold_train) if len(y_inner_fold_train) > 0 else np.full(num_classes, 1/num_classes)
                    if len(class_probs_sim_inner) != num_classes:
                        temp_probs = np.zeros(num_classes)
                        # Ensure class_probs_sim_inner indices are valid for temp_probs
                        valid_indices = np.unique(y_inner_fold_train) # Classes present in y_inner_fold_train
                        for i, cls_idx in enumerate(valid_indices):
                            if cls_idx < num_classes: # Check boundary
                                temp_probs[cls_idx] = (np.sum(y_inner_fold_train == cls_idx) / len(y_inner_fold_train)) if len(y_inner_fold_train) > 0 else 0
                        class_probs_sim_inner = temp_probs / np.sum(temp_probs) if np.sum(temp_probs) > 0 else np.full(num_classes, 1/num_classes)


                num_val_samples_inner_fold = len(y_inner_fold_val)
                dummy_stm_oof = np.random.multinomial(1, class_probs_sim_inner, size=num_val_samples_inner_fold) * 0.85 + \
                                np.random.rand(num_val_samples_inner_fold, num_classes) * 0.15
                oof_preds_inner['stm_gnn'][inner_val_idx] = dummy_stm_oof / np.sum(dummy_stm_oof, axis=1, keepdims=True)

                num_outer_test_samples = len(y_outer_test)
                dummy_stm_test = (np.random.multinomial(1, class_probs_sim_inner, size=num_outer_test_samples) * 0.85 + \
                                 np.random.rand(num_outer_test_samples, num_classes) * 0.15)
                    current_sum = np.sum(dummy_stm_test, axis=1, keepdims=True)
                    current_sum[current_sum == 0] = 1 # Avoid division by zero if all probs are zero for a sample
                    base_model_preds_on_outer_test_sum['stm_gnn'] += (dummy_stm_test / current_sum) / n_inner_folds
                    logger.info(f"Inner Fold {inner_fold_idx+1}: STM-GNN (conceptual) prediction complete.")
                except Exception as e:
                    logger.error(f"Inner Fold {inner_fold_idx+1}: Error during STM-GNN (conceptual) prediction: {e}")
                    pass # Continue


        # --- Inner CV Loop Finished for Outer Fold {outer_fold_idx+1} ---
        logger.info(f"Outer Fold {outer_fold_idx+1}: Finished generating OOF predictions from inner CV.")

        # Concatenate OOF predictions from inner loop to form training data for meta-learner
        # These OOFs are from validating on X_inner_fold_val sets.
        meta_features_train_outer_list = []
        if config.get('ensemble', {}).get('train_lgbm', True): meta_features_train_outer_list.append(oof_preds_inner['lgbm'])
        if config.get('ensemble', {}).get('train_teco', True): meta_features_train_outer_list.append(oof_preds_inner['teco'])
        if config.get('ensemble', {}).get('train_stm_gnn', True): meta_features_train_outer_list.append(oof_preds_inner['stm_gnn'])

        if not meta_features_train_outer_list:
            logger.error(f"Outer Fold {outer_fold_idx+1}: No base models were successfully trained or included for meta-learner input. Skipping meta-learner for this fold.")
            # Optionally, append NaN or skip metrics for this fold to avoid errors in averaging later
        else:
            X_meta_train_outer = np.concatenate(meta_features_train_outer_list, axis=1)
            y_meta_train_outer = y_outer_train # Labels for meta-learner are from the outer train set
            logger.info(f"Outer Fold {outer_fold_idx+1}: Meta-learner training features shape: {X_meta_train_outer.shape}")

            # --- Train XGBoost Meta-Learner on X_meta_train_outer ---
            if config.get('ensemble', {}).get('train_meta_learner', True):
                logger.info(f"Outer Fold {outer_fold_idx+1}: Training XGBoost Meta-Learner...")
                try:
            meta_config = config.get('ensemble', {}).get('meta_learner_xgb_params', {})
            xgb_meta_model_outer = XGBoostMetaLearner(
                params=meta_config.get('model_specific_params'),
                depth=meta_config.get('depth', 3)
            )
            # No validation set for meta-learner here; it's trained on all OOFs from inner loop
            # Early stopping for meta-learner in NCV needs careful consideration.
            # For now, train without it or use a small split from X_meta_train_outer if desired.
            xgb_meta_model_outer.train(
                X_meta_train_outer, y_meta_train_outer,
                num_boost_round=meta_config.get('num_boost_round', 100), # Reduced for NCV speed
                        early_stopping_rounds=None # Or a small portion of X_meta_train_outer for ES
                    )
                    logger.info(f"Outer Fold {outer_fold_idx+1}: XGBoost Meta-Learner trained.")

                    # --- Prediction with Meta-Learner on X_outer_test ---
                    # X_outer_test predictions are formed from averaged base model predictions on X_outer_test
                    meta_features_test_outer_list = []
                    if config.get('ensemble', {}).get('train_lgbm', True): meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['lgbm'])
                    if config.get('ensemble', {}).get('train_teco', True): meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['teco'])
                    if config.get('ensemble', {}).get('train_stm_gnn', True): meta_features_test_outer_list.append(base_model_preds_on_outer_test_sum['stm_gnn'])

                    X_meta_test_outer = np.concatenate(meta_features_test_outer_list, axis=1)
                    logger.debug(f"Outer Fold {outer_fold_idx+1}: Meta-learner test features shape: {X_meta_test_outer.shape}")

                    final_preds_meta_proba_outer = xgb_meta_model_outer.predict_proba(X_meta_test_outer)
                    final_preds_meta_labels_outer = xgb_meta_model_outer.predict(X_meta_test_outer)

                    # Evaluate meta-learner for this outer fold
                    acc_meta_outer = accuracy_score(y_outer_test, final_preds_meta_labels_outer)
                    f1_meta_outer = f1_score(y_outer_test, final_preds_meta_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    prec_meta_outer = precision_score(y_outer_test, final_preds_meta_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    rec_meta_outer = recall_score(y_outer_test, final_preds_meta_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    auroc_meta_outer = -1.0 # Default for error cases
                    try:
                        # Ensure correct probability column is used for binary case
                        probas_for_auc = final_preds_meta_proba_outer[:, 1] if num_classes == 2 and final_preds_meta_proba_outer.ndim == 2 and final_preds_meta_proba_outer.shape[1] >=2 else final_preds_meta_proba_outer
                        auroc_meta_outer = roc_auc_score(y_outer_test, probas_for_auc, multi_class='ovr', average='weighted')
                    except ValueError as e:
                        logger.warning(f"Outer Fold {outer_fold_idx+1} Meta AUROC calculation error: {e}. Proba shape: {final_preds_meta_proba_outer.shape}")

                    outer_fold_metrics_meta['accuracy'].append(acc_meta_outer)
                    outer_fold_metrics_meta['auroc'].append(auroc_meta_outer)
                    outer_fold_metrics_meta['f1'].append(f1_meta_outer)
                    outer_fold_metrics_meta['precision'].append(prec_meta_outer)
                    outer_fold_metrics_meta['recall'].append(rec_meta_outer)

                    wandb.log({
                        f"outer_fold_{outer_fold_idx+1}/meta_accuracy": acc_meta_outer,
                        f"outer_fold_{outer_fold_idx+1}/meta_auroc": auroc_meta_outer,
                        f"outer_fold_{outer_fold_idx+1}/meta_f1": f1_meta_outer,
                        f"outer_fold_{outer_fold_idx+1}/meta_precision": prec_meta_outer,
                        f"outer_fold_{outer_fold_idx+1}/meta_recall": rec_meta_outer,
                        "outer_fold": outer_fold_idx + 1
                    })
                    logger.info(f"Outer Fold {outer_fold_idx+1} Meta-Learner: Acc={acc_meta_outer:.4f}, AUROC={auroc_meta_outer:.4f}, F1={f1_meta_outer:.4f}")
                except Exception as e:
                    logger.error(f"Outer Fold {outer_fold_idx+1}: Error during Meta-Learner training or evaluation: {e}")
                    # Append NaNs or default values if meta-learner fails for a fold
                    for key in outer_fold_metrics_meta.keys(): outer_fold_metrics_meta[key].append(np.nan)


        # --- Soft Voting on X_outer_test (Alternative/Complementary) ---
        # This uses the averaged predictions from base models on the outer test set.
        soft_vote_weights = config.get('ensemble', {}).get('soft_vote_weights', {})
        if soft_vote_weights and any(config.get('ensemble', {}).get(f'train_{model_key}', False) for model_key in soft_vote_weights):
            logger.info(f"Outer Fold {outer_fold_idx+1}: Performing Soft Voting...")
            try:
                final_preds_soft_vote_proba_outer = np.zeros((len(y_outer_test), num_classes)) # Initialize correctly
                total_weight = 0

                active_models_for_sv = 0
                if config.get('ensemble', {}).get('train_lgbm', True) and 'lgbm' in soft_vote_weights:
                    final_preds_soft_vote_proba_outer += soft_vote_weights['lgbm'] * base_model_preds_on_outer_test_sum['lgbm']
                    total_weight += soft_vote_weights['lgbm']
                    active_models_for_sv +=1
                if config.get('ensemble', {}).get('train_teco', True) and 'teco' in soft_vote_weights:
                    final_preds_soft_vote_proba_outer += soft_vote_weights['teco'] * base_model_preds_on_outer_test_sum['teco']
                    total_weight += soft_vote_weights['teco']
                    active_models_for_sv +=1
                if config.get('ensemble', {}).get('train_stm_gnn', True) and 'stm_gnn' in soft_vote_weights:
                    final_preds_soft_vote_proba_outer += soft_vote_weights['stm_gnn'] * base_model_preds_on_outer_test_sum['stm_gnn']
                    total_weight += soft_vote_weights['stm_gnn']
                    active_models_for_sv +=1

                if active_models_for_sv > 0 and total_weight > 0: # Ensure there were models and weights
                    # Normalize probabilities if weights don't sum to 1 (or if only some models contributed)
                    # final_preds_soft_vote_proba_outer /= total_weight # Only if weights are not pre-normalized.
                    # Ensure probabilities sum to 1 per sample after weighting
                    row_sums = final_preds_soft_vote_proba_outer.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1 # Avoid division by zero if all preds are 0
                    final_preds_soft_vote_proba_outer = final_preds_soft_vote_proba_outer / row_sums

                    final_preds_soft_vote_labels_outer = np.argmax(final_preds_soft_vote_proba_outer, axis=1)

                    acc_sv_outer = accuracy_score(y_outer_test, final_preds_soft_vote_labels_outer)
                    f1_sv_outer = f1_score(y_outer_test, final_preds_soft_vote_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    prec_sv_outer = precision_score(y_outer_test, final_preds_soft_vote_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    rec_sv_outer = recall_score(y_outer_test, final_preds_soft_vote_labels_outer, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
                    auroc_sv_outer = -1.0
                    try:
                        probas_for_auc_sv = final_preds_soft_vote_proba_outer[:, 1] if num_classes == 2 and final_preds_soft_vote_proba_outer.ndim == 2 and final_preds_soft_vote_proba_outer.shape[1] >=2 else final_preds_soft_vote_proba_outer
                        auroc_sv_outer = roc_auc_score(y_outer_test, probas_for_auc_sv, multi_class='ovr', average='weighted')
                    except ValueError as e:
                        logger.warning(f"Outer Fold {outer_fold_idx+1} SoftVote AUROC calculation error: {e}. Proba shape: {final_preds_soft_vote_proba_outer.shape}")

                    outer_fold_metrics_soft_vote['accuracy'].append(acc_sv_outer)
                    outer_fold_metrics_soft_vote['auroc'].append(auroc_sv_outer)
                    outer_fold_metrics_soft_vote['f1'].append(f1_sv_outer)
                    outer_fold_metrics_soft_vote['precision'].append(prec_sv_outer)
                    outer_fold_metrics_soft_vote['recall'].append(rec_sv_outer)

                    wandb.log({
                        f"outer_fold_{outer_fold_idx+1}/sv_accuracy": acc_sv_outer,
                        f"outer_fold_{outer_fold_idx+1}/sv_auroc": auroc_sv_outer,
                        f"outer_fold_{outer_fold_idx+1}/sv_f1": f1_sv_outer,
                        # ... other sv metrics ...
                        "outer_fold": outer_fold_idx + 1
                    })
                    logger.info(f"Outer Fold {outer_fold_idx+1} Soft Vote: Acc={acc_sv_outer:.4f}, AUROC={auroc_sv_outer:.4f}, F1={f1_sv_outer:.4f}")
                else:
                    logger.warning(f"Outer Fold {outer_fold_idx+1}: Soft Voting not performed due to no active models or zero total weight.")
                    for key in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key].append(np.nan)
            except Exception as e:
                logger.error(f"Outer Fold {outer_fold_idx+1}: Error during Soft Voting: {e}")
                for key in outer_fold_metrics_soft_vote.keys(): outer_fold_metrics_soft_vote[key].append(np.nan)


    # --- Nested Cross-Validation Finished ---
    logger.info("===== Nested Cross-Validation Summary =====")
    if config.get('ensemble', {}).get('train_meta_learner', True) and len(outer_fold_metrics_meta['auroc']) > 0 :
        # Use np.nanmean and np.nanstd to handle folds where metrics might be NaN (e.g., if a model failed)
        avg_meta_acc = np.nanmean(outer_fold_metrics_meta['accuracy'])
        avg_meta_auroc = np.nanmean(outer_fold_metrics_meta['auroc'])
        std_meta_auroc = np.nanstd(outer_fold_metrics_meta['auroc'])
        avg_meta_f1 = np.nanmean(outer_fold_metrics_meta['f1'])
        avg_meta_precision = np.nanmean(outer_fold_metrics_meta['precision'])
        avg_meta_recall = np.nanmean(outer_fold_metrics_meta['recall'])

        logger.info(f"Meta-Learner Average Accuracy: {avg_meta_acc:.4f}")
        logger.info(f"Meta-Learner Average AUROC: {avg_meta_auroc:.4f} +/- {std_meta_auroc:.4f}")
        logger.info(f"Meta-Learner Average F1: {avg_meta_f1:.4f}")
        logger.info(f"Meta-Learner Average Precision: {avg_meta_precision:.4f}")
        logger.info(f"Meta-Learner Average Recall: {avg_meta_recall:.4f}")
        wandb.log({
            "ncv_summary/meta_avg_accuracy": avg_meta_acc,
            "ncv_summary/meta_avg_auroc": avg_meta_auroc,
            "ncv_summary/meta_std_auroc": std_meta_auroc,
            "ncv_summary/meta_avg_f1": avg_meta_f1,
            "ncv_summary/meta_avg_precision": avg_meta_precision,
            "ncv_summary/meta_avg_recall": avg_meta_recall,
        })

    if soft_vote_weights and sum(soft_vote_weights.values()) > 0 and len(outer_fold_metrics_soft_vote['auroc']) > 0:
        avg_sv_acc = np.nanmean(outer_fold_metrics_soft_vote['accuracy'])
        avg_sv_auroc = np.nanmean(outer_fold_metrics_soft_vote['auroc'])
        std_sv_auroc = np.nanstd(outer_fold_metrics_soft_vote['auroc'])
        avg_sv_f1 = np.nanmean(outer_fold_metrics_soft_vote['f1'])
        avg_sv_precision = np.nanmean(outer_fold_metrics_soft_vote['precision'])
        avg_sv_recall = np.nanmean(outer_fold_metrics_soft_vote['recall'])

        logger.info(f"Soft Voting Average Accuracy: {avg_sv_acc:.4f}")
        logger.info(f"Soft Voting Average AUROC: {avg_sv_auroc:.4f} +/- {std_sv_auroc:.4f}")
        logger.info(f"Soft Voting Average F1: {avg_sv_f1:.4f}")
        wandb.log({
            "ncv_summary/sv_avg_accuracy": avg_sv_acc,
            "ncv_summary/sv_avg_auroc": avg_sv_auroc,
            "ncv_summary/sv_std_auroc": std_sv_auroc,
            "ncv_summary/sv_avg_f1": avg_sv_f1,
            "ncv_summary/sv_avg_precision": avg_sv_precision,
            "ncv_summary/sv_avg_recall": avg_sv_recall,
        })


    wandb.finish()
    logger.info("Nested Cross-Validation ensemble training run finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main training script for Clinical Prediction Model.")
    parser.add_argument('--config', type=str, default='configs/dummy_train_config.yaml',
                        help='Path to the training configuration file.')
    args = parser.parse_args()

    # Create a dummy config if it doesn't exist for the example to run
    try:
        with open(args.config, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Warning: Configuration file {args.config} not found. Creating a dummy one.")
        dummy_config_content = {
            'random_seed': 42,
            'use_gpu': True,
            'dummy_data_train_samples': 500, # Reduced for faster CV example
            'dummy_data_val_samples': 100,   # This will be our "test set" for base models
            'dummy_data_features': 10,       # Features for tabular models (LGBM)
            # num_classes will be derived from dummy_data_classes for consistency in config
            'dummy_data_classes': 2,         # Binary classification for simplicity
            'dummy_data_train_weights': [0.9, 0.1], # Imbalanced
            'dummy_data_val_weights': [0.8, 0.2],   # Imbalanced

            'balancing': {
                'use_rsmote_gan_in_cv': True, # Apply RSMOTE within CV folds
                'rsmote_gan_params': { # As per spec for AUROC
                    'k': 5,
                    'minority_upsample_factor': 3.0
                }
            },
            'loss_function': { # For DL models if trained directly (not part of ensemble CV here)
                               # This section is less used by the ensemble script directly but good for full config
                'type': 'ClassBalancedFocalLoss',
                'beta': 0.9999, # As per spec for AUROC
                'gamma': 2.0    # As per spec for AUROC
            },
            'optimizer': { # For DL models
                'name': 'Adam',
                'lr': 0.001 # Default, can be tuned
            },
            'training': { # General params, less relevant for CV-based ensemble script's main flow
                'num_epochs': 5, # Default, can be tuned (used by conceptual DL model training if any)
            },
            'ensemble': {
                'n_folds_for_oof': 5, # As per spec for meta-learner
                'train_lgbm': True,
                'train_teco': True,  # Will use dummy predictions
                'train_stm_gnn': True, # Will use dummy predictions
                'train_meta_learner': True,
                'lgbm_params': { # As per spec for AUROC
                    'num_leaves': 10000,
                    'class_weight': 'balanced',
                    'num_boost_round': 1000, # Default, can be tuned
                    'early_stopping_rounds': 50, # Default, can be tuned
                    # 'model_specific_params': {'learning_rate': 0.05} # Example for custom internal lgbm params
                    'save_path': 'lgbm_final.joblib' # Path to save the final model trained on full data (conceptual)
                },
                'teco_params': { # As per spec for AUROC
                     # Conceptual parameters, assuming TECOTransformerModel takes these
                    'input_feature_dim': 10, # Placeholder, should match actual data feature dim for TECO
                    'd_model': 512,
                    'num_encoder_layers': 4,
                    'nhead': 8, # Typical for d_model=512
                    'dim_feedforward': 2048, # Typical for d_model=512
                    'dropout': 0.1,
                    'num_classes': 2, # Placeholder, should match dummy_data_classes
                    'max_seq_len': 100, # Example, should match data
                    'save_path': 'teco_final.pth' # Conceptual save path
                },
                'stm_gnn_params': { # As per spec for AUROC
                    # Conceptual parameters, assuming STMGNN or ModelWithSTMGNNLayer takes these
                    'num_node_features': 10, # Placeholder, should match actual data
                    'layer_hidden_dim': 256, # For STMGNNLayer consistency
                    'gnn_output_dim': 256,   # For STMGNNLayer consistency
                    'num_gnn_layers': 5,
                    'global_memory_dim': 128,
                    'num_memory_slots': 10, # Example
                    'num_heads': 8,
                    'dropout': 0.1,
                    'num_classes': 2, # Placeholder, should match dummy_data_classes
                    'save_path': 'stm_gnn_final.pth' # Conceptual save path
                },
                'meta_learner_xgb_params': { # As per spec for AUROC
                    'depth': 3,
                    'num_boost_round': 200, # Default, can be tuned
                    'early_stopping_rounds': 20, # Default, can be tuned
                    # 'model_specific_params': {'eta': 0.05} # Example for custom internal xgb params
                    'save_path': 'meta_learner_xgb.joblib'
                },
                'soft_vote_weights': { # As per spec for AUROC
                    'stm_gnn': 0.5,
                    'lgbm': 0.3,
                    'teco': 0.2
                }
            }
        }
        # Update placeholders based on other config values for consistency
        dummy_config_content['ensemble']['teco_params']['input_feature_dim'] = dummy_config_content['dummy_data_features']
        dummy_config_content['ensemble']['teco_params']['num_classes'] = dummy_config_content['dummy_data_classes']
        dummy_config_content['ensemble']['stm_gnn_params']['num_node_features'] = dummy_config_content['dummy_data_features']
        dummy_config_content['ensemble']['stm_gnn_params']['num_classes'] = dummy_config_content['dummy_data_classes']

        # Ensure configs directory exists
        import os
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(dummy_config_content, f, default_flow_style=False)
        print(f"Created dummy config at {args.config}")

    main(args.config)
