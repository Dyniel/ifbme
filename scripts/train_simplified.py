import sys
import os
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_utils.data_loader import load_raw_data
from data_utils.preprocess import get_preprocessor
from models.lgbm_model import LightGBMModel
from utils.metrics import dt_score_calc, ls_score_calc

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    config = load_config(config_path)

    # Load data
    X_full_raw_df, y_full_raw_series = load_raw_data(config, base_data_path=config.get('data_dir', 'data/'))

    # Encode target variable
    le = LabelEncoder()
    y_full_encoded = le.fit_transform(y_full_raw_series)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Split data
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full_raw_df, y_full_encoded, test_size=0.2, random_state=config.get('random_seed', 42), stratify=y_full_encoded
    )

    # Preprocess data
    preproc_cfg = config.get('preprocessing', {})
    numerical_cols = preproc_cfg.get('numerical_cols', [])
    categorical_cols = preproc_cfg.get('categorical_cols', [])

    preprocessor = get_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        imputation_strategy=preproc_cfg.get('imputation_strategy', 'median'),
        scale_numerics=preproc_cfg.get('scale_numerics', True),
        handle_unknown_categorical=preproc_cfg.get('onehot_handle_unknown', 'ignore')
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    # --- Train Discharge Type Model ---
    dt_config = config.get('discharge_type_model', {})

    def dt_objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'class_weight': 'balanced'
        }

        model = LightGBMModel(params=params)
        model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
        preds = model.predict(X_val)

        death_label = class_mapping.get('Death', 0)
        f1 = f1_score(y_val, preds, pos_label=death_label, average='binary')
        return f1

    study_dt = optuna.create_study(direction='maximize')
    study_dt.optimize(dt_objective, n_trials=dt_config.get('n_trials', 50))

    best_params_dt = study_dt.best_params
    dt_model = LightGBMModel(params=best_params_dt)
    dt_model.train(X_train, y_train)

    # --- Train Length of Stay Model ---
    los_config = config.get('length_of_stay_model', {})
    y_los_train = X_train_raw[config.get('los_column', 'LengthOfStay')]
    y_los_val = X_val_raw[config.get('los_column', 'LengthOfStay')]

    def los_objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        model = LightGBMModel(params=params)
        model.train(X_train, y_los_train, X_val, y_los_val, early_stopping_rounds=50)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_los_val, preds)
        return mae

    study_los = optuna.create_study(direction='minimize')
    study_los.optimize(los_objective, n_trials=los_config.get('n_trials', 50))

    best_params_los = study_los.best_params
    los_model = LightGBMModel(params=best_params_los)
    los_model.train(X_train, y_los_train)

    # Save models
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    dt_model.save_model(os.path.join(output_dir, 'dt_model.joblib'))
    los_model.save_model(os.path.join(output_dir, 'los_model.joblib'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(class_mapping, os.path.join(output_dir, 'class_mapping.joblib'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simplified training script for IUPESM 2025 challenge.")
    parser.add_argument('--config', type=str, default='configs/simplified_train_config.yaml',
                        help='Path to the training configuration file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(1)

    main(args.config)
