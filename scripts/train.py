import argparse
import yaml # For loading configurations
import torch
import torch.optim as optim
import numpy as np

# Project-specific imports (will be refined)
# from data_utils.loader import YourDataLoader # Replace with actual data loader
from data_utils.balancing import RSMOTEGAN
from data_utils.losses import ClassBalancedFocalLoss
# from models.your_model import YourModel # Replace with actual model

def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_samples_per_class(y_train):
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

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu")
    print(f"Using device: {device}")

    # Seed for reproducibility
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Data Loading and Preprocessing ---
    print("Conceptual: Loading data...")
    # Replace with actual data loading logic
    # X_train_raw, y_train_raw = YourDataLoader(config['data']['train_path'], ...).load()
    # X_val_raw, y_val_raw = YourDataLoader(config['data']['val_path'], ...).load()

    # Dummy data for now
    num_samples_train = config.get('dummy_data_train_samples', 1000)
    num_samples_val = config.get('dummy_data_val_samples', 200)
    num_features = config.get('dummy_data_features', 20)
    num_classes = config.get('dummy_data_classes', 2)

    # Create imbalanced dummy data
    weights_train = config.get('dummy_data_train_weights', [0.9, 0.1])
    # Ensure weights sum to 1 for np.random.choice
    p_train = np.array(weights_train) / np.sum(weights_train)
    y_train_raw = np.random.choice(num_classes, num_samples_train, p=p_train)
    X_train_raw = np.random.rand(num_samples_train, num_features)

    weights_val = config.get('dummy_data_val_weights', [0.9, 0.1])
    p_val = np.array(weights_val) / np.sum(weights_val)
    y_val_raw = np.random.choice(num_classes, num_samples_val, p=p_val)
    X_val_raw = np.random.rand(num_samples_val, num_features)

    print(f"Original training data shape: X={X_train_raw.shape}, y={y_train_raw.shape}")
    print(f"Original training class distribution: {np.bincount(y_train_raw)}")


    # --- Data Balancing (RSMOTE-GAN) ---
    if config.get('balancing', {}).get('use_rsmote_gan', False):
        print("Applying RSMOTE-GAN...")
        rsmote_config = config['balancing'].get('rsmote_gan_params', {})
        rsmote_gan = RSMOTEGAN(
            k_neighbors=rsmote_config.get('k', 5),
            minority_upsample_factor=rsmote_config.get('minority_upsample_factor', 3.0),
            random_state=seed
        )
        X_train, y_train = rsmote_gan.fit_resample(X_train_raw, y_train_raw)
        print(f"Resampled training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Resampled training class distribution: {np.bincount(y_train)}")
    else:
        X_train, y_train = X_train_raw, y_train_raw

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val_raw, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_raw, dtype=torch.long).to(device)


    # --- Model Definition ---
    print("Conceptual: Defining model...")
    # model = YourModel(input_dim=X_train_tensor.shape[1], **config['model_params']).to(device)
    # Dummy model for now
    model = torch.nn.Linear(num_features, num_classes).to(device) # Example: Simple Linear model
    print(model)

    # --- Loss Function (Class-Balanced Focal Loss) ---
    loss_config = config.get('loss_function', {})
    if loss_config.get('type') == 'ClassBalancedFocalLoss':
        print("Using Class-Balanced Focal Loss.")
        # Get samples per class from the original (before balancing) or resampled training labels
        # The paper "Class-Balanced Loss Based on Effective Number of Samples" calculates weights
        # based on the original label distribution.
        # If using RSMOTE-GAN, the distribution is already altered.
        # For this example, let's use the distribution of y_train_raw if balancing is applied,
        # or y_train if no balancing, to determine the 'natural' frequencies for weighting.
        # The choice here depends on the strategy: balance via data, via loss, or both.
        # The prompt says "RSMOTE/GAN-SMOTE + Class-Balanced Focal Loss", suggesting both.
        # The CBFL paper suggests using original distribution for calculating weights.

        samples_per_class_for_loss = get_samples_per_class(y_train_raw) # Based on original distribution
        print(f"Samples per class (for CBFL weights): {samples_per_class_for_loss}")

        criterion = ClassBalancedFocalLoss(
            beta=loss_config.get('beta', 0.9999),
            gamma=loss_config.get('gamma', 2.0),
            samples_per_class=samples_per_class_for_loss,
            reduction='mean'
        )
    else:
        print("Using standard CrossEntropyLoss.")
        criterion = torch.nn.CrossEntropyLoss()
    print(criterion)

    # --- Optimizer ---
    optimizer_config = config.get('optimizer', {})
    opt_name = optimizer_config.get('name', 'Adam')
    lr = optimizer_config.get('lr', 1e-3)

    if opt_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=optimizer_config.get('momentum', 0.9))
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
    print(optimizer)

    # --- Model Training ---
    # This will now be more complex to handle base models and meta-learner for ensemble.
    # The plan specifies: "Soft-vote (0.5 STM-GNN / 0.3 LGBM / 0.2 TECO) + meta-learner XGBoost"
    # This usually means:
    # 1. Train base models.
    # 2. Get their predictions (often OOF for meta-learner training data).
    # 3. For final prediction: either soft-vote OR use meta-learner. The "+" suggests both might be options
    #    or meta-learner is trained on OOF, and final output is meta-learner's output.
    #    Let's assume meta-learner is the primary way to combine, trained on OOF.
    #    The soft-vote weights (0.5/0.3/0.2) might be an alternative way to combine, or perhaps
    #    they are feature weights if base model outputs are concatenated before meta-learner.
    #    The spec "Soft-vote (0.5 / 0.3 / 0.2) + meta-learner podbija zwykle +0,02 AUROC"
    #    suggests these are two separate strategies or soft-vote is an input to meta.
    #    Let's assume meta-learner trains on OOF predictions of base models.

    print("Starting ensemble training process...")

    # Convert all data to PyTorch tensors for deep learning models first
    # (LGBM/XGBoost wrappers will handle numpy/pandas)
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    # X_val_tensor = torch.tensor(X_val_raw, dtype=torch.float32).to(device) # Using X_val_raw for DL model val
    # y_val_tensor = torch.tensor(y_val_raw, dtype=torch.long).to(device)


    # --- Cross-validation for generating OOF predictions for Meta-Learner ---
    n_folds = config.get('ensemble', {}).get('n_folds_for_oof', 5)
    # Use original X_train_raw, y_train_raw for CV splitting before any initial balancing
    # Balancing should happen INSIDE the CV fold if it's data-dependent.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Placeholders for OOF predictions from each base model
    # These will store predictions on the validation part of each fold.
    # Dimensions: (num_train_samples, num_classes_base_model_output)
    # For LGBM/XGBoost, this is num_classes. For DL models, also num_classes (after softmax).

    # Need to know num_classes for sizing OOF arrays
    # num_classes determined by y_train_raw

    oof_preds = {
        'lgbm': np.zeros((len(y_train_raw), num_classes)), # Assuming num_classes is known
        'teco': np.zeros((len(y_train_raw), num_classes)),
        'stm_gnn': np.zeros((len(y_train_raw), num_classes))
    }

    # Placeholders for test predictions from each base model (averaged over folds)
    test_preds_sum = {
        'lgbm': np.zeros((len(y_val_raw), num_classes)), # Using y_val_raw as stand-in for test set size
        'teco': np.zeros((len(y_val_raw), num_classes)),
        'stm_gnn': np.zeros((len(y_val_raw), num_classes))
    }

    # Actual test set (using X_val_raw, y_val_raw as a proxy for a true test set here)
    X_test_for_base_models = X_val_raw
    y_test_for_base_models = y_val_raw


    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train_raw)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        X_fold_train, y_fold_train = X_train_raw[train_idx], y_train_raw[train_idx]
        X_fold_val, y_fold_val = X_train_raw[val_idx], y_train_raw[val_idx]

        # --- Optional: Apply RSMOTE-GAN to (X_fold_train, y_fold_train) ---
        if config.get('balancing', {}).get('use_rsmote_gan_in_cv', True): # New config flag
            print(f"Fold {fold+1}: Applying RSMOTE-GAN to fold training data...")
            rsmote_cv_config = config['balancing'].get('rsmote_gan_params', {})
            rsmote_gan_cv = RSMOTEGAN(
                k_neighbors=rsmote_cv_config.get('k', 5),
                minority_upsample_factor=rsmote_cv_config.get('minority_upsample_factor', 3.0),
                random_state=seed + fold # Vary seed per fold
            )
            X_fold_train_balanced, y_fold_train_balanced = rsmote_gan_cv.fit_resample(X_fold_train, y_fold_train)
            print(f"Fold {fold+1}: Resampled training data shape: X={X_fold_train_balanced.shape}")
        else:
            X_fold_train_balanced, y_fold_train_balanced = X_fold_train, y_fold_train


        # --- 1. Train LightGBM on (X_fold_train_balanced, y_fold_train_balanced) ---
        if config.get('ensemble', {}).get('train_lgbm', True):
            print(f"\nFold {fold+1}: Training LightGBM...")
            lgbm_config = config.get('ensemble', {}).get('lgbm_params', {})
            lgbm_fold_model = LightGBMModel(
                params=lgbm_config.get('model_specific_params'), # e.g., learning_rate, n_estimators
                num_leaves=lgbm_config.get('num_leaves', 10000),
                class_weight=lgbm_config.get('class_weight', 'balanced')
            )
            lgbm_fold_model.train(X_fold_train_balanced, y_fold_train_balanced,
                                  X_fold_val, y_fold_val, # Use original fold val for early stopping
                                  num_boost_round=lgbm_config.get('num_boost_round', 1000),
                                  early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 50))

            oof_preds['lgbm'][val_idx] = lgbm_fold_model.predict_proba(X_fold_val)
            test_preds_sum['lgbm'] += lgbm_fold_model.predict_proba(X_test_for_base_models) / n_folds
            # lgbm_fold_model.save_model(f"lgbm_fold_{fold+1}.joblib") # Optional: save fold model

        # --- 2. Train TECO-Transformer (Conceptual) ---
        if config.get('ensemble', {}).get('train_teco', True):
            print(f"\nFold {fold+1}: Training TECO-Transformer (Conceptual)...")
            # This requires sequence data. Assuming X_fold_train_balanced can be shaped into sequences.
            # Or, specific sequence data loader is needed. For now, conceptual.
            # teco_config = config.get('ensemble', {}).get('teco_params', {})
            # teco_model = TECOTransformerModel(...)
            # ... training loop for TECO ...
            # oof_preds['teco'][val_idx] = teco_model.predict_proba(X_fold_val_sequences)
            # test_preds_sum['teco'] += teco_model.predict_proba(X_test_sequences) / n_folds
            # For dummy run, fill with random predictions based on class balance
            num_val_samples_fold = len(y_fold_val)
            # Simulate probabilities respecting original class balance for more realistic dummy OOF
            class_probs_sim = np.bincount(y_train_raw) / len(y_train_raw)
            oof_preds['teco'][val_idx] = np.random.multinomial(1, class_probs_sim, size=num_val_samples_fold) * 0.8 + \
                                         np.random.rand(num_val_samples_fold, num_classes) * 0.2 # Add noise
            oof_preds['teco'][val_idx] = oof_preds['teco'][val_idx] / np.sum(oof_preds['teco'][val_idx], axis=1, keepdims=True)


            num_test_samples_fold = len(y_test_for_base_models)
            test_preds_sum['teco'] += (np.random.multinomial(1, class_probs_sim, size=num_test_samples_fold) * 0.8 + \
                                      np.random.rand(num_test_samples_fold, num_classes) * 0.2) / n_folds
            test_preds_sum['teco'] = np.clip(test_preds_sum['teco'], 0, 1) # Ensure valid probs after averaging


        # --- 3. Train STM-GNN (Conceptual) ---
        if config.get('ensemble', {}).get('train_stm_gnn', True):
            print(f"\nFold {fold+1}: Training STM-GNN (Conceptual)...")
            # This requires graph snapshot data. Assuming X_fold_train_balanced can be used/transformed.
            # Or, specific graph data loader is needed. For now, conceptual.
            # stm_gnn_config = config.get('ensemble', {}).get('stm_gnn_params', {})
            # stm_gnn_model = STMGNN(...)
            # ... training loop for STM-GNN ...
            # oof_preds['stm_gnn'][val_idx] = stm_gnn_model.predict_proba(X_fold_val_graphs)
            # test_preds_sum['stm_gnn'] += stm_gnn_model.predict_proba(X_test_graphs) / n_folds
            # For dummy run:
            class_probs_sim = np.bincount(y_train_raw) / len(y_train_raw)
            num_val_samples_fold = len(y_fold_val)
            oof_preds['stm_gnn'][val_idx] = np.random.multinomial(1, class_probs_sim, size=num_val_samples_fold) * 0.85 + \
                                            np.random.rand(num_val_samples_fold, num_classes) * 0.15 # Add noise
            oof_preds['stm_gnn'][val_idx] = oof_preds['stm_gnn'][val_idx] / np.sum(oof_preds['stm_gnn'][val_idx], axis=1, keepdims=True)

            num_test_samples_fold = len(y_test_for_base_models)
            test_preds_sum['stm_gnn'] += (np.random.multinomial(1, class_probs_sim, size=num_test_samples_fold) * 0.85 + \
                                         np.random.rand(num_test_samples_fold, num_classes) * 0.15) / n_folds
            test_preds_sum['stm_gnn'] = np.clip(test_preds_sum['stm_gnn'], 0, 1)


    print("\n--- Finished generating OOF predictions for base models ---")
    # At this point, oof_preds contains predictions from each base model for the entire training set.
    # These will be the features for the meta-learner.

    # Concatenate OOF predictions to form training data for meta-learner
    meta_features_train_list = []
    if config.get('ensemble', {}).get('train_lgbm', True):
        meta_features_train_list.append(oof_preds['lgbm'])
    if config.get('ensemble', {}).get('train_teco', True):
        meta_features_train_list.append(oof_preds['teco'])
    if config.get('ensemble', {}).get('train_stm_gnn', True):
        meta_features_train_list.append(oof_preds['stm_gnn'])

    if not meta_features_train_list:
        raise ValueError("No base models were trained or included for meta-learner input.")

    X_meta_train = np.concatenate(meta_features_train_list, axis=1)
    y_meta_train = y_train_raw # Meta-learner is trained on original labels

    print(f"Meta-learner training features shape: {X_meta_train.shape}") # (num_train_samples, num_base_models * num_classes)

    # --- Train XGBoost Meta-Learner ---
    if config.get('ensemble', {}).get('train_meta_learner', True):
        print("\n--- Training XGBoost Meta-Learner ---")
        meta_config = config.get('ensemble', {}).get('meta_learner_xgb_params', {})
        xgb_meta_model = XGBoostMetaLearner(
            params=meta_config.get('model_specific_params'),
            depth=meta_config.get('depth', 3)
        )
        # For meta-learner, we might not have a separate validation set easily if all data used for OOF.
        # Could split X_meta_train or use a portion of it. For now, train on all OOF.
        # Or, could perform another CV loop for meta-learner training (complex).
        # Simplest: train on all OOF, use early stopping if a small val split is made from OOF.
        xgb_meta_model.train(X_meta_train, y_meta_train,
                             num_boost_round=meta_config.get('num_boost_round', 200),
                             early_stopping_rounds=meta_config.get('early_stopping_rounds', 20) # Requires validation set
                            )
        # xgb_meta_model.save_model(config['ensemble']['meta_learner_save_path'])
        print("XGBoost Meta-Learner trained.")

        # --- Final Prediction using Meta-Learner (on the proxy test set) ---
        # Create test features for meta-learner
        meta_features_test_list = []
        if config.get('ensemble', {}).get('train_lgbm', True):
            meta_features_test_list.append(test_preds_sum['lgbm']) # Averaged test preds
        if config.get('ensemble', {}).get('train_teco', True):
            meta_features_test_list.append(test_preds_sum['teco'])
        if config.get('ensemble', {}).get('train_stm_gnn', True):
            meta_features_test_list.append(test_preds_sum['stm_gnn'])

        X_meta_test = np.concatenate(meta_features_test_list, axis=1)

        final_predictions_meta_proba = xgb_meta_model.predict_proba(X_meta_test)
        final_predictions_meta_labels = xgb_meta_model.predict(X_meta_test)
        print(f"Meta-learner predictions on test set (proxy) - Proba shape: {final_predictions_meta_proba.shape}")

        # Evaluate meta-learner (example)
        # accuracy_meta = accuracy_score(y_test_for_base_models, final_predictions_meta_labels)
        # print(f"Meta-Learner Test Accuracy (proxy): {accuracy_meta:.4f}")

    # --- Soft Voting (Alternative/Complementary) ---
    # Weights: STM-GNN 0.5, LightGBM 0.3, TECO-Transformer 0.2
    soft_vote_weights = config.get('ensemble', {}).get('soft_vote_weights', {})
    if soft_vote_weights:
        print("\n--- Performing Soft Voting (on proxy test set predictions) ---")
        final_predictions_soft_vote_proba = np.zeros_like(test_preds_sum['lgbm']) # Init with one shape

        total_weight = 0
        if config.get('ensemble', {}).get('train_lgbm', True) and 'lgbm' in soft_vote_weights:
            final_predictions_soft_vote_proba += soft_vote_weights['lgbm'] * test_preds_sum['lgbm']
            total_weight += soft_vote_weights['lgbm']
        if config.get('ensemble', {}).get('train_teco', True) and 'teco' in soft_vote_weights:
            final_predictions_soft_vote_proba += soft_vote_weights['teco'] * test_preds_sum['teco']
            total_weight += soft_vote_weights['teco']
        if config.get('ensemble', {}).get('train_stm_gnn', True) and 'stm_gnn' in soft_vote_weights:
            final_predictions_soft_vote_proba += soft_vote_weights['stm_gnn'] * test_preds_sum['stm_gnn']
            total_weight += soft_vote_weights['stm_gnn']

        if total_weight > 0:
             # Normalize if weights don't sum to 1 (though they should ideally)
            # final_predictions_soft_vote_proba /= total_weight
            final_predictions_soft_vote_labels = np.argmax(final_predictions_soft_vote_proba, axis=1)
            print(f"Soft-vote predictions on test set (proxy) - Proba shape: {final_predictions_soft_vote_proba.shape}")
            # accuracy_soft_vote = accuracy_score(y_test_for_base_models, final_predictions_soft_vote_labels)
            # print(f"Soft-Vote Test Accuracy (proxy): {accuracy_soft_vote:.4f}")
        else:
            print("No valid models/weights for soft voting.")


    print("\nEnsemble training conceptual run finished.")


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

```
