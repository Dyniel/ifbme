import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import time

from data_utils import load_and_preprocess_data
from models import MortalityPredictor, LoSPredictor

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data paths
TRAIN_CSV = 'data/trainData.csv'
VAL_CSV = 'data/valData.csv'

# Model Hyperparameters (can be tuned)
DIM_HIDDEN = 128  # Hidden dimension for GraphGPS
NUM_GPS_LAYERS = 3
NUM_ATTN_HEADS = 4
LAP_PE_K_DIM = 8    # Dimension for Laplacian PE (k eigenvectors)
SIGN_PE_K_DIM = 0   # Dimension for SignNet features (set to 0 if not using/available)
DROPOUT_RATE = 0.2 # Increased dropout slightly

# Training Hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100 # Max epochs, early stopping will likely intervene
PATIENCE_EARLY_STOPPING = 15
COSINE_T_0 = 10 # For CosineAnnealingWarmRestarts: Number of iterations for the first restart.
COSINE_T_MULT = 2 # For CosineAnnealingWarmRestarts: A factor increases T_i after a restart.
WARMUP_EPOCHS = 5

# File names for saved models
MORTALITY_MODEL_PATH = 'model_dt.pt'
LOS_MODEL_PATH = 'model_los.pt'

# --- Helper Functions ---
def get_input_dim_from_data(sample_csv_path, lap_pe_k_dim, sign_pe_k_dim):
    """
    Infers the input dimension by preprocessing a small sample of data.
    This is crucial because the number of features can change based on one-hot encoding.
    """
    print("Inferring input dimension from sample data...")
    # Load a small part of data to get feature dimension after preprocessing
    # No need to fit preprocessor here if it's already done, but we need its output shape
    # For safety, we'll create a throwaway preprocessor on a small slice if one isn't provided
    # However, the main function will fit it on full train and pass it.
    # This function is more of a utility if you need dim_in *before* full data loading.

    # Re-using the main data loading logic for consistency
    graph_data_sample, _, preprocessor_sample = load_and_preprocess_data(
        sample_csv_path,
        fit_preprocessor=True, # Fit a temporary preprocessor
        target_cols=None,      # Not needed for dim inference
        k_neighbors=3          # Small k, not relevant for feature dim
    )
    if graph_data_sample is None or preprocessor_sample is None:
        raise RuntimeError("Failed to process sample data for dimension inference.")

    dim_in = graph_data_sample.x.shape[1]
    print(f"Inferred input dimension (features + PEs): {dim_in}") # This dim already includes PE from AddLaplacianEigenvectorPE

    # The current data_utils.py adds LapPE *after* ColumnTransformer.
    # The model's atom_encoder expects raw features, PEs are added separately inside the model.
    # So, we need the dimension *before* PEs are added by the transform, then add PE dims in model constructor.

    # Let's get the dimension from the preprocessor directly
    num_numerical_features = len(preprocessor_sample.transformers_[0][2])

    cat_transformer = preprocessor_sample.named_transformers_['cat']
    onehot_encoder = cat_transformer.named_steps['onehot']
    num_onehot_features = 0
    if hasattr(onehot_encoder, 'get_feature_names_out'):
        num_onehot_features = len(onehot_encoder.get_feature_names_out(preprocessor_sample.transformers_[1][2]))
    elif hasattr(onehot_encoder, 'categories_'):
        for cats in onehot_encoder.categories_:
            num_onehot_features += len(cats)

    dim_processed_features = num_numerical_features + num_onehot_features
    print(f"Dimension from preprocessor (numerical + one-hot categorical): {dim_processed_features}")
    return dim_processed_features


def train_model(model, train_data, val_data, criterion, optimizer, scheduler,
                task_name, epochs, patience, model_path):
    best_val_metric = -np.inf if task_name == "Mortality" else np.inf
    epochs_no_improve = 0

    print(f"\n--- Starting Training: {task_name} ---")
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    print(f"Criterion: {criterion}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        # Assuming train_data is a single Data object for the entire training graph
        out = model(train_data.to(DEVICE))

        if task_name == "Mortality":
            target = train_data.y_mortality.to(DEVICE)
            loss = criterion(out, target)
        elif task_name == "LoS":
            # Apply log transform to target for LoS as specified
            target = torch.log1p(train_data.y_los.to(DEVICE)) # log1p for stability (log(x+1))
            loss = criterion(out, target)
        else:
            raise ValueError("Unknown task name")

        loss.backward()
        optimizer.step()

        if scheduler and epoch > WARMUP_EPOCHS: # Apply scheduler after warmup
             if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch - WARMUP_EPOCHS)
             else: # For other schedulers like ReduceLROnPlateau
                # Evaluation needed for ReduceLROnPlateau, handled below
                pass

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.to(DEVICE))
            if task_name == "Mortality":
                val_target = val_data.y_mortality.to(DEVICE)
                val_loss = criterion(val_out, val_target)
                # Apply sigmoid for AUROC calculation
                val_probs = torch.sigmoid(val_out).cpu().numpy()
                val_metric = roc_auc_score(val_target.cpu().numpy(), val_probs)
                metric_name = "AUROC"
            elif task_name == "LoS":
                val_target_log = torch.log1p(val_data.y_los.to(DEVICE))
                val_loss = criterion(val_out, val_target_log)
                # Inverse transform for RMSE: exp(preds) - 1
                val_preds_original_scale = torch.expm1(val_out).cpu().numpy()
                val_target_original_scale = val_data.y_los.cpu().numpy()
                val_metric = np.sqrt(mean_squared_error(val_target_original_scale, val_preds_original_scale.clip(min=0)))
                metric_name = "RMSE"

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val {metric_name}: {val_metric:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_duration:.2f}s")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric if task_name == "Mortality" else val_loss) # AUROC higher is better, RMSE lower is better

        # Early stopping and model saving
        if task_name == "Mortality": # Higher is better for AUROC
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), model_path)
                print(f"Improved {metric_name}. Model saved to {model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else: # Lower is better for RMSE
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), model_path)
                print(f"Improved {metric_name}. Model saved to {model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print(f"Finished training for {task_name}. Best Val {metric_name}: {best_val_metric:.4f}")
    model.load_state_dict(torch.load(model_path)) # Load best model
    return model

# --- Main Training Execution ---
if __name__ == "__main__":
    print("Starting main training script execution...")

    # Infer input dimension from data (IMPORTANT: ensure this matches model's dim_in)
    # This requires preprocessing a small part of the data to know the number of features
    # after one-hot encoding etc.
    # The `dim_in` for the GraphGPS model should be the number of node features *before* LapPE/SignNet
    # are added within the model, as the model has separate encoders for them.

    # We need to load data first to get the preprocessor and the shape of X *after* ColumnTransformer
    print("Loading and preprocessing training data for dimension inference and training...")
    train_graph, train_targets_df, preprocessor = load_and_preprocess_data(
        TRAIN_CSV,
        fit_preprocessor=True,
        target_cols=['outcomeType', 'lengthofStay'],
        k_neighbors=10 # As per document
    )
    if train_graph is None:
        print("Failed to load training data. Exiting.")
        exit()

    # The number of features from the ColumnTransformer output
    # This is before LapPE is added by the PyG transform if AddLaplacianEigenvectorPE is used *after*
    # But our current data_utils.py applies AddLaplacianEigenvectorPE *inside* load_and_preprocess_data
    # and stores it as 'lap_pe'. The model then uses this attribute.
    # So, data.x from data_utils should be the features *before* PEs are explicitly added by model.
    # The model's atom_encoder will take data.x
    # The model's pe_encoders will take data.lap_pe and data.sign_pe

    # Let's adjust dim_in to be the number of features from the preprocessor,
    # and the model internally handles adding PE dimensions.

    # Get the number of features output by the ColumnTransformer
    # This is the actual input dimension to the model's self.atom_encoder
    num_numerical_features_fitted = len(preprocessor.transformers_[0][2])
    cat_transformer_fitted = preprocessor.named_transformers_['cat']
    onehot_encoder_fitted = cat_transformer_fitted.named_steps['onehot']
    num_onehot_features_fitted = 0
    if hasattr(onehot_encoder_fitted, 'get_feature_names_out'):
        num_onehot_features_fitted = len(onehot_encoder_fitted.get_feature_names_out(preprocessor.transformers_[1][2]))
    elif hasattr(onehot_encoder_fitted, 'categories_'): # Fallback for older sklearn
        for cats in onehot_encoder_fitted.categories_:
            num_onehot_features_fitted += len(cats)

    DIM_IN_FEATURES_FROM_DATA = num_numerical_features_fitted + num_onehot_features_fitted
    print(f"Actual input dimension from preprocessed data (excluding explicit PEs): {DIM_IN_FEATURES_FROM_DATA}")

    # Check if lap_pe was added by data_utils and its dimension
    actual_lap_pe_dim = 0
    if hasattr(train_graph, 'lap_pe') and train_graph.lap_pe is not None:
        actual_lap_pe_dim = train_graph.lap_pe.shape[1]
        print(f"LapPE found in data with dimension: {actual_lap_pe_dim}")
        if actual_lap_pe_dim != LAP_PE_K_DIM:
            print(f"WARNING: LAP_PE_K_DIM ({LAP_PE_K_DIM}) in train script config differs from actual LapPE dim ({actual_lap_pe_dim}) in data. Using actual: {actual_lap_pe_dim}")
            # LAP_PE_K_DIM = actual_lap_pe_dim # Use the dimension from data
    else:
        print("No LapPE found in data. lap_pe_dim will be effectively 0 for the model if not provided.")
        LAP_PE_K_DIM = 0 # Ensure model knows no LapPE is coming if not in data.

    # SIGN_PE_K_DIM is already 0 by default. If it were > 0, we'd need data.sign_pe

    print("Loading and preprocessing validation data...")
    val_graph, val_targets_df = load_and_preprocess_data(
        VAL_CSV,
        preprocessor=preprocessor,
        fit_preprocessor=False,
        target_cols=['outcomeType', 'lengthofStay'],
        k_neighbors=10
    )
    if val_graph is None:
        print("Failed to load validation data. Exiting.")
        exit()

    # --- Train Mortality Model ---
    print("\nInitializing Mortality Predictor...")
    mortality_model = MortalityPredictor(
        dim_in=DIM_IN_FEATURES_FROM_DATA,  # Features from ColumnTransformer
        dim_h=DIM_HIDDEN,
        num_layers=NUM_GPS_LAYERS,
        num_heads=NUM_ATTN_HEADS,
        lap_pe_dim=actual_lap_pe_dim, # Dimension of LapPE provided in Data object
        sign_pe_dim=SIGN_PE_K_DIM,  # Dimension of SignNet PE provided in Data object
        dropout=DROPOUT_RATE
    ).to(DEVICE)

    # Weighted BCE loss for imbalanced classification
    # Calculate pos_weight for mortality
    if train_graph.y_mortality is not None:
        num_positive = torch.sum(train_graph.y_mortality == 1)
        num_negative = torch.sum(train_graph.y_mortality == 0)
        pos_weight_value = num_negative / (num_positive + 1e-6) # Add epsilon to avoid division by zero
        print(f"Mortality: Negatives={num_negative}, Positives={num_positive}, Pos_weight={pos_weight_value:.2f}")
        criterion_mortality = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=DEVICE))
    else:
        print("Warning: y_mortality not found in train_graph. Using unweighted BCEWithLogitsLoss.")
        criterion_mortality = nn.BCEWithLogitsLoss()

    optimizer_mortality = optim.AdamW(mortality_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_mortality = CosineAnnealingWarmRestarts(optimizer_mortality, T_0=COSINE_T_0, T_mult=COSINE_T_MULT)

    mortality_model = train_model(mortality_model, train_graph, val_graph, criterion_mortality,
                                  optimizer_mortality, scheduler_mortality, "Mortality",
                                  EPOCHS, PATIENCE_EARLY_STOPPING, MORTALITY_MODEL_PATH)

    # --- Train LoS Model ---
    print("\nInitializing LoS Predictor...")
    los_model = LoSPredictor(
        dim_in=DIM_IN_FEATURES_FROM_DATA, # Features from ColumnTransformer
        dim_h=DIM_HIDDEN,
        num_layers=NUM_GPS_LAYERS,
        num_heads=NUM_ATTN_HEADS,
        lap_pe_dim=actual_lap_pe_dim, # Dimension of LapPE provided in Data object
        sign_pe_dim=SIGN_PE_K_DIM,  # Dimension of SignNet PE provided in Data object
        dropout=DROPOUT_RATE
    ).to(DEVICE)

    criterion_los = nn.MSELoss() # Target will be log-transformed
    optimizer_los = optim.AdamW(los_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_los = CosineAnnealingWarmRestarts(optimizer_los, T_0=COSINE_T_0, T_mult=COSINE_T_MULT)

    los_model = train_model(los_model, train_graph, val_graph, criterion_los,
                            optimizer_los, scheduler_los, "LoS",
                            EPOCHS, PATIENCE_EARLY_STOPPING, LOS_MODEL_PATH)

    print("\nTraining finished. Models saved to:")
    print(f"Mortality Model: {MORTALITY_MODEL_PATH}")
    print(f"LoS Model: {LOS_MODEL_PATH}")
