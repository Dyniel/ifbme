import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix
import numpy as np
import pandas as pd
import os
import time
import wandb  # Dodano wandb
import yaml  # Do wczytywania konfiguracji sweepa
import argparse  # Do obsługi argumentów linii poleceń dla sweepów
import io
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc, confusion_matrix as sk_confusion_matrix

from data_utils import load_and_preprocess_data
from models import MortalityPredictor, LoSPredictor

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight # For compatibility with BCEWithLogitsLoss usage pattern

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)  # Prevents nans when BCE_loss is large
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

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
LAP_PE_K_DIM = 8  # Dimension for Laplacian PE (k eigenvectors)
SIGN_PE_K_DIM = 0  # Dimension for SignNet features (set to 0 if not using/available)
DROPOUT_RATE = 0.2

# Training Hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100  # Max epochs, early stopping will likely intervene
PATIENCE_EARLY_STOPPING = 15
COSINE_T_0 = 10
COSINE_T_MULT = 2
WARMUP_EPOCHS = 5

# File names for saved models
MORTALITY_MODEL_PATH = 'model_dt.pt'
LOS_MODEL_PATH = 'model_los.pt'

# Wandb configuration
WANDB_PROJECT = "ifbme-projekt"
WANDB_ENTITY = None  # Uzupełnij, jeśli używasz teamu w wandb
USE_WANDB = True # Default to using wandb, can be overridden by CLI


# --- Helper Functions ---
def get_input_dim_from_data(sample_csv_path, lap_pe_k_dim, sign_pe_k_dim):
    """
    Infers the input dimension by preprocessing a small sample of data.
    """
    print("Inferring input dimension from sample data...")
    graph_data_sample, _, preprocessor_sample = load_and_preprocess_data(
        sample_csv_path,
        fit_preprocessor=True,
        target_cols=None,
        k_neighbors=3
    )
    if graph_data_sample is None or preprocessor_sample is None:
        raise RuntimeError("Failed to process sample data for dimension inference.")

    num_numerical_features = 0
    if 'num' in preprocessor_sample.named_transformers_ and preprocessor_sample.named_transformers_['num'] != 'drop':
        num_numerical_features = len(preprocessor_sample.transformers_[0][2])

    num_onehot_features = 0
    if 'cat' in preprocessor_sample.named_transformers_ and preprocessor_sample.named_transformers_['cat'] != 'drop':
        cat_transformer = preprocessor_sample.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']
        if hasattr(onehot_encoder, 'get_feature_names_out'):
            try:
                num_onehot_features = len(onehot_encoder.get_feature_names_out(preprocessor_sample.transformers_[1][2]))
            except Exception:  # Fallback if categorical features were not present in sample
                num_onehot_features = 0
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

        out = model(train_data.to(DEVICE))

        if task_name == "Mortality":
            target = train_data.y_mortality.to(DEVICE)
            if _LABEL_SMOOTHING > 0.0 and isinstance(criterion, nn.BCEWithLogitsLoss):
                # Apply label smoothing to targets
                # For binary classification:
                # target = 0 becomes 0 + label_smoothing / 2
                # target = 1 becomes 1 - label_smoothing + label_smoothing / 2 = 1 - label_smoothing / 2
                # Simplified: target * (1.0 - label_smoothing) + 0.5 * label_smoothing
                # No, for BCE, it's: y_ls = y_true * (1 - eps) + eps / K where K is num_classes (2 for binary)
                # So for y=0, y_ls = 0 * (1-eps) + eps/2 = eps/2
                # For y=1, y_ls = 1 * (1-eps) + eps/2 = 1 - eps + eps/2 = 1 - eps/2
                # This means (1-target) * eps/2 + target * (1-eps/2)
                target = target.float() # ensure float
                target = (1.0 - target) * (_LABEL_SMOOTHING / 2.0) + target * (1.0 - _LABEL_SMOOTHING / 2.0)

            loss = criterion(out, target)
        elif task_name == "LoS":
            target = torch.log1p(train_data.y_los.to(DEVICE))
            loss = criterion(out, target)
        else:
            raise ValueError("Unknown task name")

        loss.backward()
        if _CLIP_GRAD_NORM > 0: # Apply gradient clipping if configured
            torch.nn.utils.clip_grad_norm_(model.parameters(), _CLIP_GRAD_NORM)
        optimizer.step()

        if scheduler and epoch > WARMUP_EPOCHS:
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch - WARMUP_EPOCHS)
            else:
                pass

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.to(DEVICE))

            # Initialize a dictionary to hold all data for this epoch's wandb.log call
            log_data_for_epoch = {}

            if task_name == "Mortality":
                val_target = val_data.y_mortality.to(DEVICE)
                val_loss = criterion(val_out, val_target)
                val_probs_np = torch.sigmoid(val_out).cpu().numpy().flatten()  # Spłaszczenie
                val_target_cpu_np = val_target.cpu().numpy().flatten()  # Spłaszczenie
                val_metric = roc_auc_score(val_target_cpu_np, val_probs_np)
                metric_name = "AUROC"

                if USE_WANDB and wandb.run is not None:
                    try:
                        # --- Ręczne generowanie krzywej ROC ---
                        fpr, tpr, _ = sk_roc_curve(val_target_cpu_np, val_probs_np)
                        roc_auc_value = sk_auc(fpr, tpr)
                        plt.figure()
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'Receiver Operating Characteristic - {task_name}')
                        plt.legend(loc="lower right")
                        buf_roc = io.BytesIO()
                        plt.savefig(buf_roc, format='png')
                        buf_roc.seek(0)
                        roc_image = Image.open(buf_roc)
                        log_data_for_epoch[f"{task_name}/roc_curve_image"] = wandb.Image(roc_image)
                        plt.close()
                        buf_roc.close()

                        # --- Ręczne generowanie macierzy konfuzji ---
                        val_preds_classes_np = (val_probs_np > 0.5).astype(int)
                        cm = sk_confusion_matrix(val_target_cpu_np, val_preds_classes_np)
                        fig_cm, ax_cm = plt.subplots()
                        cax = ax_cm.matshow(cm, cmap=plt.cm.Blues)
                        fig_cm.colorbar(cax)
                        class_names = ['Survival', 'Death']
                        ax_cm.set_xticks(np.arange(len(class_names)))
                        ax_cm.set_yticks(np.arange(len(class_names)))
                        ax_cm.set_xticklabels(class_names)
                        ax_cm.set_yticklabels(class_names)
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'Confusion Matrix - {task_name}')
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax_cm.text(j, i, str(cm[i, j]), va='center', ha='center',
                                           color='black' if cm[i, j] < cm.max() / 2 else 'white')
                        buf_cm = io.BytesIO()
                        plt.savefig(buf_cm, format='png')
                        buf_cm.seek(0)
                        cm_image = Image.open(buf_cm)
                        log_data_for_epoch[f"{task_name}/confusion_matrix_image"] = wandb.Image(cm_image)
                        plt.close(fig_cm)
                        buf_cm.close()
                    except Exception as e:
                        print(f"Error preparing wandb plots for Mortality: {e}")

            elif task_name == "LoS":
                val_target_log = torch.log1p(val_data.y_los.to(DEVICE))
                val_loss = criterion(val_out, val_target_log)
                val_preds_original_scale_np = torch.expm1(val_out).cpu().numpy().clip(min=0).flatten()
                val_target_original_scale_np = val_data.y_los.cpu().numpy().flatten()
                val_metric = np.sqrt(mean_squared_error(val_target_original_scale_np, val_preds_original_scale_np))
                metric_name = "RMSE"

                if USE_WANDB and wandb.run is not None:
                    try:
                        data_scatter_list = [[float(true_val), float(pred_val)] for true_val, pred_val in
                                             zip(val_target_original_scale_np, val_preds_original_scale_np)]
                        table_scatter = wandb.Table(data=data_scatter_list, columns=["Actual LoS", "Predicted LoS"])
                        log_data_for_epoch[f"{task_name}/predictions_scatter"] = wandb.plot.scatter(
                            table_scatter, "Actual LoS", "Predicted LoS", title="Actual vs. Predicted LoS"
                        )
                    except Exception as e:
                        print(f"Error preparing wandb plots for LoS: {e}")

        epoch_duration = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val {metric_name}: {val_metric:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_duration:.2f}s")

        if USE_WANDB and wandb.run is not None:
            # Add scalar metrics to the dictionary
            log_data_for_epoch.update({
                f"{task_name}/train_loss": loss.item(),
                f"{task_name}/val_loss": val_loss.item(),
                f"{task_name}/val_{metric_name.lower()}": val_metric,
                f"{task_name}/learning_rate": optimizer.param_groups[0]['lr'],
                f"{task_name}/epoch_duration_sec": epoch_duration,
            })
            # Single log call for the epoch
            if log_data_for_epoch:  # Ensure there's something to log
                wandb.log(log_data_for_epoch, step=epoch)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric if task_name == "Mortality" else val_loss)

        if task_name == "Mortality":
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), model_path)
                print(f"Improved {metric_name}. Model saved to {model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
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
    model.load_state_dict(torch.load(model_path))

    if USE_WANDB and wandb.run is not None:
        wandb.save(model_path)
        wandb.summary[f"best_val_{metric_name.lower()}_{task_name.lower()}"] = best_val_metric
    return model, best_val_metric  # Zwracamy również najlepszą metrykę


# --- Main Training Execution ---
def main(run_config_from_sweep=None):
    """
    Główna funkcja trenująca modele.
    Args:
        run_config_from_sweep (dict, optional): Słownik konfiguracji z wandb sweep.
    """
    if run_config_from_sweep is None:
        # Use command-line args for manual tuning if provided, otherwise script defaults
        # This block is for non-sweep runs
        current_run_config = {
            "learning_rate": args.lr if hasattr(args, 'lr') else LEARNING_RATE,
            "dim_hidden": DIM_HIDDEN, # Keep other defaults or make them CLI args too if needed
            "num_gps_layers": NUM_GPS_LAYERS,
            "num_attn_heads": NUM_ATTN_HEADS,
            "dropout_rate": args.dropout if hasattr(args, 'dropout') else DROPOUT_RATE,
            "lap_pe_k_dim": LAP_PE_K_DIM,
            "sign_pe_k_dim": SIGN_PE_K_DIM,
            "weight_decay": args.wd if hasattr(args, 'wd') else WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience_early_stopping": PATIENCE_EARLY_STOPPING,
            "cosine_t_0": COSINE_T_0,
            "cosine_t_mult": COSINE_T_MULT,
            "warmup_epochs": WARMUP_EPOCHS,
            "activation_fn": args.activation_fn if hasattr(args, 'activation_fn') else 'relu',
            "clip_grad_norm": args.clip_grad_norm if hasattr(args, 'clip_grad_norm') else 0.0,
            "label_smoothing": args.label_smoothing if hasattr(args, 'label_smoothing') else 0.0,
            # "batch_size": args.batch_size if hasattr(args, 'batch_size') else 64 # If batch_size CLI arg is added
        }
        wandb_mode = "online" # or "disabled" if --no-wandb is also handled here
        # Prosta nazwa dla pojedynczego uruchomienia, wandb doda unikalne ID
        run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        wandb_mode = "disabled" if not USE_WANDB else "online"
    else: # This is a sweep run
        current_run_config = run_config_from_sweep
        wandb_mode = "online"  # Sweep agent always uses wandb online
        run_name = f"sweep_run_{wandb.run.id if wandb.run else time.strftime('%Y%m%d-%H%M%S')}" # wandb.run should exist for agent

    effective_config = current_run_config

    if USE_WANDB:
        if wandb.run is None: # Only init if not already initialized by agent
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config=current_run_config,  # Pass current_run_config, wandb makes a copy
                name=run_name,
                mode=wandb_mode # This will be 'online' for sweeps or default runs, 'disabled' if --no-wandb
            )
        # If wandb.init was called (either here or by agent), wandb.config will be populated
        # For sweep runs, wandb.config is the source of truth provided by the controller.
        # For single runs, wandb.config is a copy of current_run_config.
        if wandb.run: # Check if init was successful or if run is active
             effective_config = wandb.config # Use wandb's config as the primary source
    else:
        print("Weights & Biases is disabled. Using local configuration.")
        # effective_config is already current_run_config

    # Pobierz wartości z effective_config, using .get() for safety, especially for optional params
    _LEARNING_RATE = effective_config.get("learning_rate", LEARNING_RATE)
    _DIM_HIDDEN = effective_config.get("dim_hidden", DIM_HIDDEN)
    _NUM_GPS_LAYERS = effective_config.get("num_gps_layers", NUM_GPS_LAYERS)
    _NUM_ATTN_HEADS = effective_config.get("num_attn_heads", NUM_ATTN_HEADS)
    _DROPOUT_RATE = effective_config.get("dropout_rate", DROPOUT_RATE)
    _LAP_PE_K_DIM = effective_config.get("lap_pe_k_dim", LAP_PE_K_DIM)
    _SIGN_PE_K_DIM = effective_config.get("sign_pe_k_dim", SIGN_PE_K_DIM)
    _WEIGHT_DECAY = effective_config.get("weight_decay", WEIGHT_DECAY)
    _EPOCHS = effective_config.get("epochs", EPOCHS)
    _PATIENCE_EARLY_STOPPING = effective_config.get("patience_early_stopping", PATIENCE_EARLY_STOPPING)
    _COSINE_T_0 = effective_config.get("cosine_t_0", COSINE_T_0)
    _COSINE_T_MULT = effective_config.get("cosine_t_mult", COSINE_T_MULT)
    _WARMUP_EPOCHS = effective_config.get("warmup_epochs", WARMUP_EPOCHS)
    _BATCH_SIZE_PARAM = effective_config.get("batch_size", 64)
    _ACTIVATION_FN = effective_config.get("activation_fn", "relu")
    _CLIP_GRAD_NORM = effective_config.get("clip_grad_norm", 0.0)
    _LABEL_SMOOTHING = effective_config.get("label_smoothing", 0.0)

    print("Starting main training script execution...")
    print(f"Running with effective_config: {effective_config}")
    # Note: _BATCH_SIZE_PARAM is available here but not directly used in the current GNN full-graph processing.
    # It's made available for future modifications if subgraph batching is implemented.

    print("Loading and preprocessing training data for dimension inference and training...")
    train_graph, _, preprocessor = load_and_preprocess_data(
        TRAIN_CSV, fit_preprocessor=True, target_cols=['outcomeType', 'lengthofStay'], k_neighbors=10
    )
    if train_graph is None:
        print("Failed to load training data. Exiting.")
        if USE_WANDB and wandb.run: wandb.finish(exit_code=1)
        exit()

    num_numerical_features_fitted = 0
    if 'num' in preprocessor.named_transformers_ and preprocessor.named_transformers_['num'] != 'drop':
        num_numerical_features_fitted = len(preprocessor.transformers_[0][2])

    num_onehot_features_fitted = 0
    if 'cat' in preprocessor.named_transformers_ and preprocessor.named_transformers_['cat'] != 'drop':
        cat_transformer_fitted = preprocessor.named_transformers_['cat']
        onehot_encoder_fitted = cat_transformer_fitted.named_steps['onehot']
        if hasattr(onehot_encoder_fitted, 'get_feature_names_out'):
            try:  # Handle cases where categorical_features might be empty
                num_onehot_features_fitted = len(
                    onehot_encoder_fitted.get_feature_names_out(preprocessor.transformers_[1][2]))
            except Exception:
                num_onehot_features_fitted = 0  # If no cat features were passed to onehot
        elif hasattr(onehot_encoder_fitted, 'categories_'):
            for cats in onehot_encoder_fitted.categories_:
                num_onehot_features_fitted += len(cats)

    DIM_IN_FEATURES_FROM_DATA = num_numerical_features_fitted + num_onehot_features_fitted
    print(f"Actual input dimension from preprocessed data (excluding explicit PEs): {DIM_IN_FEATURES_FROM_DATA}")

    actual_lap_pe_dim = 0
    if hasattr(train_graph, 'lap_pe') and train_graph.lap_pe is not None:
        actual_lap_pe_dim = train_graph.lap_pe.shape[1]
        print(f"LapPE found in data with dimension: {actual_lap_pe_dim}")
        if actual_lap_pe_dim != _LAP_PE_K_DIM:
            print(
                f"WARNING: Configured LAP_PE_K_DIM ({_LAP_PE_K_DIM}) differs from actual LapPE dim in data ({actual_lap_pe_dim}). Using actual: {actual_lap_pe_dim} for model init.")
            # Model should use actual_lap_pe_dim; wandb.config logs the intended _LAP_PE_K_DIM
    else:
        print("No LapPE found in data. LapPE dim for model will be 0.")

    actual_sign_pe_dim = 0
    if hasattr(train_graph, 'sign_pe') and train_graph.sign_pe is not None:
        actual_sign_pe_dim = train_graph.sign_pe.shape[1]
        print(f"SignPE found in data with dimension: {actual_sign_pe_dim}")
        if actual_sign_pe_dim != _SIGN_PE_K_DIM:
            print(
                f"WARNING: Configured SIGN_PE_K_DIM ({_SIGN_PE_K_DIM}) differs from actual SignPE dim in data ({actual_sign_pe_dim}). Using actual: {actual_sign_pe_dim} for model init.")
    elif _SIGN_PE_K_DIM > 0:
        print(
            f"WARNING: SIGN_PE_K_DIM is {_SIGN_PE_K_DIM} but no SignPE found in data. SignPE dim for model will be 0.")

    print("Loading and preprocessing validation data...")
    val_graph, _ = load_and_preprocess_data(
        VAL_CSV, preprocessor=preprocessor, fit_preprocessor=False,
        target_cols=['outcomeType', 'lengthofStay'], k_neighbors=10
    )
    if val_graph is None:
        print("Failed to load validation data. Exiting.")
        if USE_WANDB and wandb.run: wandb.finish(exit_code=1)
        exit()

    # --- Train Mortality Model ---
    print("\nInitializing Mortality Predictor...")
    mortality_model = MortalityPredictor(
        dim_in=DIM_IN_FEATURES_FROM_DATA, dim_h=_DIM_HIDDEN, num_layers=_NUM_GPS_LAYERS,
        num_heads=_NUM_ATTN_HEADS, lap_pe_dim=actual_lap_pe_dim,  # Use actual dim from data
        sign_pe_dim=actual_sign_pe_dim, dropout=_DROPOUT_RATE,  # Use actual dim from data
        activation_fn_str=_ACTIVATION_FN
    ).to(DEVICE)

    if USE_WANDB and wandb.run:
        wandb.config.update({
            "mortality_model_architecture": str(mortality_model),
            "input_features_dim_from_data": DIM_IN_FEATURES_FROM_DATA,
            "actual_lap_pe_dim_from_data_for_model": actual_lap_pe_dim,  # Log what model actually uses
            "actual_sign_pe_dim_from_data_for_model": actual_sign_pe_dim  # Log what model actually uses
        }, allow_val_change=True)

    if train_graph.y_mortality is not None:
        num_positive = torch.sum(train_graph.y_mortality == 1)
        num_negative = torch.sum(train_graph.y_mortality == 0)
        pos_weight_value = num_negative / (num_positive + 1e-6)
        print(f"Mortality: Negatives={num_negative}, Positives={num_positive}, Pos_weight={pos_weight_value:.2f}")

        loss_fn_choice = args.loss_fn if hasattr(args, 'loss_fn') else 'bce' # Default to bce if not in args (e.g. during sweep)
        if loss_fn_choice == 'focal':
            print("Using FocalLoss for mortality task.")
            # Alpha can be tuned, common values are 0.25 for positive class if it's rare, or 1-alpha for negative.
            # Here, pos_weight is also passed for consistency, though alpha also addresses imbalance.
            criterion_mortality = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=torch.tensor([pos_weight_value], device=DEVICE))
        else: # Default to BCEWithLogitsLoss
            print(f"Using BCEWithLogitsLoss for mortality task with label smoothing: {_LABEL_SMOOTHING}")
            criterion_mortality = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight_value], device=DEVICE),
            )
            # For label smoothing with BCEWithLogitsLoss, we adjust targets directly if label_smoothing > 0
            # The loss function itself in PyTorch doesn't have a direct label_smoothing param like nn.CrossEntropyLoss
            # So, this will be handled during target creation or by a custom wrapper if a more standard way is needed.
            # For now, we'll note that PyTorch's BCEWithLogitsLoss expects raw logits and hard targets (0 or 1).
            # True label smoothing requires adjusting target values (e.g., 0 -> eps/N, 1 -> 1 - eps + eps/N).
            # We will adjust targets if _LABEL_SMOOTHING > 0 before passing to criterion.
            # This adjustment will be done in the train_model loop.
    else:
        print("Warning: y_mortality not found in train_graph. Using unweighted loss.")
        loss_fn_choice = args.loss_fn if hasattr(args, 'loss_fn') else 'bce'
        if loss_fn_choice == 'focal':
            criterion_mortality = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            print(f"Using BCEWithLogitsLoss (unweighted) with label smoothing: {_LABEL_SMOOTHING if _LABEL_SMOOTHING > 0 else 'N/A'}")
            criterion_mortality = nn.BCEWithLogitsLoss()


    optimizer_mortality = optim.AdamW(mortality_model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    scheduler_mortality = CosineAnnealingWarmRestarts(optimizer_mortality, T_0=_COSINE_T_0, T_mult=_COSINE_T_MULT)

    mortality_model, best_auroc_mortality = train_model(mortality_model, train_graph, val_graph, criterion_mortality,
                                                        optimizer_mortality, scheduler_mortality, "Mortality",
                                                        _EPOCHS, _PATIENCE_EARLY_STOPPING, MORTALITY_MODEL_PATH)
    if USE_WANDB and wandb.run:
        wandb.summary["final_best_auroc_mortality"] = best_auroc_mortality

    # --- Train LoS Model ---
    print("\nInitializing LoS Predictor...")
    los_model = LoSPredictor(
        dim_in=DIM_IN_FEATURES_FROM_DATA, dim_h=_DIM_HIDDEN, num_layers=_NUM_GPS_LAYERS,
        num_heads=_NUM_ATTN_HEADS, lap_pe_dim=actual_lap_pe_dim,  # Use actual dim from data
        sign_pe_dim=actual_sign_pe_dim, dropout=_DROPOUT_RATE,  # Use actual dim from data
        activation_fn_str=_ACTIVATION_FN
    ).to(DEVICE)

    if USE_WANDB and wandb.run:
        wandb.config.update({"los_model_architecture": str(los_model)}, allow_val_change=True)

    criterion_los = nn.MSELoss()
    optimizer_los = optim.AdamW(los_model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    scheduler_los = CosineAnnealingWarmRestarts(optimizer_los, T_0=_COSINE_T_0, T_mult=_COSINE_T_MULT)

    los_model, best_rmse_los = train_model(los_model, train_graph, val_graph, criterion_los,
                                           optimizer_los, scheduler_los, "LoS",
                                           _EPOCHS, _PATIENCE_EARLY_STOPPING, LOS_MODEL_PATH)
    if USE_WANDB and wandb.run:
        wandb.summary["final_best_rmse_los"] = best_rmse_los

    print("\nTraining finished. Models saved to:")
    print(f"Mortality Model: {MORTALITY_MODEL_PATH}")
    print(f"LoS Model: {LOS_MODEL_PATH}")

    # Obliczanie i logowanie GLscore
    if USE_WANDB and wandb.run is not None:
        # Normalizacja RMSE (prosta, można dostosować)
        # Chcemy, aby wyższy wynik był lepszy, więc dla RMSE (gdzie niższy jest lepszy),
        # możemy użyć 1 - znormalizowany_rmse.
        # Normalizacja RMSE do zakresu ~0-1: normalized_rmse = rmse / (C + rmse), gdzie C to stała, np. 1 lub średnie RMSE.
        # Lub prostsze: exp(-k * rmse)
        # Dla tej implementacji użyjemy: 1 / (1 + RMSE) jako "wynik" dla LoS, aby był w (0,1] i wyższy był lepszy.
        if best_rmse_los is not None and best_auroc_mortality is not None:
            los_score_component = 1 / (1 + best_rmse_los)  # Wyższy jest lepszy, zakres (0,1) dla RMSE > 0
            gl_score = best_auroc_mortality + los_score_component

            wandb.summary["los_score_component"] = los_score_component
            wandb.summary["GLscore"] = gl_score
            print(
                f"GLscore calculated: {gl_score:.4f} (AUROC: {best_auroc_mortality:.4f}, LoS_component: {los_score_component:.4f} from RMSE: {best_rmse_los:.4f})")
        else:
            print("Could not calculate GLscore because one or both primary metrics are missing.")

        if USE_WANDB and wandb.run: # Ensure wandb.run exists before trying to finish it
            wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None, help='Wandb sweep ID to run an agent for.')
    parser.add_argument('--sweep_config', type=str, default='sweep.yaml', help='Path to the sweep configuration file.')
    parser.add_argument('--project', type=str, default=WANDB_PROJECT, help='Wandb project name.')
    parser.add_argument('--entity', type=str, default=WANDB_ENTITY, help='Wandb entity (user or team).')
    parser.add_argument('--count', type=int, default=None, help='Number of runs for the sweep agent.')

    # Arguments for manual hyperparameter tuning
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--wd', type=float, default=WEIGHT_DECAY, help=f'Weight decay (default: {WEIGHT_DECAY})')
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE, help=f'Dropout rate (default: {DROPOUT_RATE})')
    # Note: Batch size is complex for current GNN setup, added as a placeholder in config if needed later.
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce', 'focal'], help='Loss function for mortality task (bce or focal, default: bce)')
    parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'gelu', 'leaky_relu'], help='Activation function for GNN layers (default: relu)')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Max norm for gradient clipping (0.0 to disable, default: 0.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor for BCE loss (default: 0.0, no smoothing)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging.')


    args = parser.parse_args()

    if args.no_wandb:
        global USE_WANDB
        USE_WANDB = False
        print("Weights & Biases logging explicitly disabled via --no-wandb flag.")

    if args.sweep_id:
        if not USE_WANDB:
            print("WARNING: --no-wandb is set, but sweep functionality relies on W&B. W&B will be enabled for sweep agent.")
            global USE_WANDB # Force enable for sweep
            USE_WANDB = True
        print(f"Starting wandb agent for sweep_id: {args.sweep_id}")


        # Funkcja `main` oczekuje argumentu `run_config_from_sweep`, ale wandb.agent przekaże config bezpośrednio.
        # Aby to pogodzić, możemy stworzyć prostą funkcję opakowującą (wrapper),
        # która przyjmie config od agenta i przekaże go do `main` pod oczekiwaną nazwą.
        # Jednakże, wandb.agent jest wystarczająco inteligentny, by przekazać config do funkcji,
        # która go akceptuje jako pierwszy argument lub jako argument nazwany 'config'.
        # Zmieniono nazwę argumentu w main na `run_config_from_sweep` dla jasności,
        # ale agent powinien sobie z tym poradzić. Jeśli nie, potrzebny byłby wrapper.
        # Dla pewności, agent wandb przekazuje config jako kwargs, więc `main` musi akceptować `**kwargs`
        # lub mieć argument o nazwie `config`. Zmieniam `main` tak, by akceptował `config` jako argument.

        # Re-definiujemy `main_for_sweep` aby pasował do oczekiwań `wandb.agent`
        # który przekazuje konfigurację jako pojedynczy argument słownikowy.
        def main_for_sweep(config_from_agent=None):
            main(run_config_from_sweep=config_from_agent)


        wandb.agent(sweep_id=args.sweep_id, function=main_for_sweep, count=args.count, project=args.project,
                    entity=args.entity)
    else:
        print("Starting a single training run (not a sweep agent).")
        main()  # Wywołanie z config=None, użyje domyślnych wartości