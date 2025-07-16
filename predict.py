#!/usr/bin/env python
# predict.py

import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_utils.sequence_loader import TabularSequenceDataset, basic_collate_fn

def main():
    parser = argparse.ArgumentParser(
        description="Make predictions with our ensemble + TECO + LoS models"
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help="Path to your saved model bundle (best_model.joblib)"
    )
    parser.add_argument(
        '--test_csv', type=str, required=True,
        help="CSV of test data (must have exactly the same raw feature columns you trained on)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="Batch size for TECO inference"
    )
    args = parser.parse_args()

    # 1) Load your artifacts
    bundle            = joblib.load(args.model_path)
    preprocessor      = bundle['preprocessor']
    lgbm_model        = bundle['lgbm_model']
    teco_model        = bundle['teco_model']
    los_regressor     = bundle['los_reg']
    soft_vote_weights = bundle.get('soft_vote_weights', {})
    thresholds        = bundle.get('thresholds', {})

    # 2) Read raw test data
    df = pd.read_csv(args.test_csv)

    # 3) If lengthofStay was part of your training features but isn't here, inject it so the ColumnTransformer sees the same columns
    if 'lengthofStay' not in df.columns:
        df['lengthofStay'] = np.nan

    # 4) Find which columns your numeric→median pipeline expects, and coerce them to float
    #    so that median imputer will see only numeric data.
    numeric_cols = []
    for name, transformer, cols in preprocessor.transformers_:
        # If you used a Pipeline for numerics:
        if isinstance(transformer, Pipeline):
            # assume first step is the imputer
            step = transformer.steps[0][1]
            if isinstance(step, SimpleImputer) and step.strategy == 'median':
                numeric_cols = cols
        # Or if you attached a raw SimpleImputer:
        elif isinstance(transformer, SimpleImputer) and transformer.strategy == 'median':
            numeric_cols = cols

    # Now coerce those columns into numbers
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5) Preprocess
    X_proc = preprocessor.transform(df)

    # 6) LightGBM probabilities
    raw_lgbm = lgbm_model.predict_proba(X_proc)
    # force into shape (N,2)
    if raw_lgbm.ndim == 1:
        probs_lgbm = np.vstack([1 - raw_lgbm, raw_lgbm]).T
    elif raw_lgbm.shape[1] == 1:
        probs_lgbm = np.hstack([1 - raw_lgbm, raw_lgbm])
    else:
        probs_lgbm = raw_lgbm

    # 7) TECO probabilities
    #    TECOTransformerModel doesn’t have predict_proba, so we run forward + softmax
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_proc.shape[1])]

    df_teco = pd.DataFrame(X_proc, columns=feature_names)
    dummy_y = np.zeros(len(df_teco), dtype=int)
    teco_ds = TabularSequenceDataset(
        df_teco, dummy_y, feature_names, target_column_name="dummy"
    )
    loader = DataLoader(
        teco_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=basic_collate_fn
    )

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teco_model = teco_model.to(device).eval()

    pieces = []
    with torch.no_grad():
        for batch in loader:
            seq   = batch["sequence"].to(device)
            mask  = batch["padding_mask"].to(device)
            logits= teco_model(seq, mask)
            probs = torch.softmax(logits, dim=1)
            pieces.append(probs.cpu().numpy())

    raw_teco = np.vstack(pieces)
    if raw_teco.ndim == 1:
        probs_teco = np.vstack([1 - raw_teco, raw_teco]).T
    elif raw_teco.shape[1] == 1:
        probs_teco = np.hstack([1 - raw_teco, raw_teco])
    else:
        probs_teco = raw_teco

    # 8) Soft‐vote ensemble
    N, C = probs_lgbm.shape
    final_probas = np.zeros((N, C), dtype=float)
    total_w      = 0.0

    for name, probs in (("lgbm", probs_lgbm), ("teco", probs_teco)):
        w = soft_vote_weights.get(name, 0)
        if w > 0 and probs.shape == final_probas.shape:
            final_probas += w * probs
            total_w      += w

    if total_w > 0:
        final_probas /= total_w
        # normalize each row to sum to 1
        rowsum = final_probas.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1
        final_probas = final_probas / rowsum

    # 9) Apply your 'death_best' threshold (default .5)
    thr     = thresholds.get("death_best", 0.5)
    p_death = final_probas[:, 0]
    preds   = np.where(p_death > thr, "Death", "Survival")

    # 10) Length‐of‐Stay prediction
    los_trans = los_regressor.predict(X_proc)
    los_pred  = np.expm1(los_trans)
    los_pred  = np.maximum(0, los_pred)

    # 11) Write out
    pd.DataFrame({"predicted_outcome": preds}) \
      .to_csv("DTestimation.csv", index=False)

    pd.DataFrame({"Estimated_LS": los_pred}) \
      .to_csv("LSestimation.csv", index=False)

    print("Saved → DTestimation.csv and LSestimation.csv")

if __name__ == "__main__":
    main()