import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.metrics import dt_score_calc, ls_score_calc, gl_score_calc


def main(gt_path, dt_pred_path, ls_pred_path):
    # Load data
    gt_df = pd.read_csv(gt_path)
    dt_pred = pd.read_csv(dt_pred_path).values.flatten()
    ls_pred = pd.read_csv(ls_pred_path).values.flatten()

    # Get ground truth values
    y_dt_true = gt_df['Outcome']
    y_ls_true = gt_df['LengthOfStay']

    # Encode DT ground truth
    le = LabelEncoder()
    y_dt_true_encoded = le.fit_transform(y_dt_true)

    # Calculate DT score
    f1 = f1_score(y_dt_true_encoded, dt_pred, average='binary', pos_label=le.transform(['Death'])[0])
    dt_score = dt_score_calc(f1)

    # Calculate LS score
    mae = mean_absolute_error(y_ls_true, ls_pred)
    ls_score = ls_score_calc(mae)

    # Calculate GL score
    gl_score = gl_score_calc(dt_score, ls_score)

    print(f"DT Score: {dt_score:.4f}")
    print(f"LS Score: {ls_score:.4f}")
    print(f"GL Score: {gl_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script for IUPESM 2025 challenge.")
    parser.add_argument('--gt', type=str, default='data/valData.csv',
                        help='Path to the ground truth data file.')
    parser.add_argument('--dt_pred', type=str, default='predictions/DTestimation.csv',
                        help='Path to the discharge type prediction file.')
    parser.add_argument('--ls_pred', type=str, default='predictions/LSestimation.csv',
                        help='Path to the length of stay prediction file.')
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        sys.exit(1)

    if not os.path.exists(args.dt_pred):
        sys.exit(1)

    if not os.path.exists(args.ls_pred):
        sys.exit(1)

    main(args.gt, args.dt_pred, args.ls_pred)
