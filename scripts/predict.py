import sys
import os
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.lgbm_model import LightGBMModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path, test_data_path, output_dir):
    config = load_config(config_path)

    # Load models and preprocessor
    model_dir = config.get('output_dir', 'outputs')
    dt_model = LightGBMModel()
    dt_model.load_model(os.path.join(model_dir, 'dt_model.joblib'))
    los_model = LightGBMModel()
    los_model.load_model(os.path.join(model_dir, 'los_model.joblib'))
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    class_mapping = joblib.load(os.path.join(model_dir, 'class_mapping.joblib'))

    # Load test data
    test_df = pd.read_csv(test_data_path)
    X_test = preprocessor.transform(test_df)

    # Predict discharge type
    dt_preds_proba = dt_model.predict_proba(X_test)
    death_label = class_mapping.get('Death', 0)
    # This assumes 'Death' is the positive class. A threshold of 0.5 is used here.
    # For the competition, it might be beneficial to tune this threshold on a validation set.
    dt_preds = (dt_preds_proba[:, death_label] > 0.5).astype(int)

    # Predict length of stay
    los_preds = los_model.predict(X_test).round(0).astype(int)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "DTestimation.csv"), dt_preds, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(output_dir, "LSestimation.csv"), los_preds, fmt='%d', delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction script for IUPESM 2025 challenge.")
    parser.add_argument('--config', type=str, default='configs/simplified_train_config.yaml',
                        help='Path to the training configuration file.')
    parser.add_argument('--test_data', type=str, default='data/testData.csv',
                        help='Path to the test data file.')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save the prediction files.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(1)

    if not os.path.exists(args.test_data):
        sys.exit(1)

    main(args.config, args.test_data, args.output_dir)
