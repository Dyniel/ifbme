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
import wandb # Dodano wandb
import yaml # Do wczytywania konfiguracji sweepa
import argparse # Do obsługi argumentów linii poleceń dla sweepów
import io
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc, confusion_matrix as sk_confusion_matrix


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
DROPOUT_RATE = 0.2

# Training Hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100 # Max epochs, early stopping will likely intervene
PATIENCE_EARLY_STOPPING = 15
COSINE_T_0 = 10
COSINE_T_MULT = 2
WARMUP_EPOCHS = 5

# File names for saved models
MORTALITY_MODEL_PATH = 'model_dt.pt'
LOS_MODEL_PATH = 'model_los.pt'

# Wandb configuration
WANDB_PROJECT = "ifbme-projekt"
WANDB_ENTITY = None # Uzupełnij, jeśli używasz teamu w wandb


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
            except Exception: # Fallback if categorical features were not present in sample
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
            loss = criterion(out, target)
        elif task_name == "LoS":
            target = torch.log1p(train_data.y_los.to(DEVICE))
            loss = criterion(out, target)
        else:
            raise ValueError("Unknown task name")

        loss.backward()
        optimizer.step()

        if scheduler and epoch > WARMUP_EPOCHS:
             if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch - WARMUP_EPOCHS)
             else:
                pass

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.to(DEVICE))
            log_plots_dict = {}
            if task_name == "Mortality":
                val_target = val_data.y_mortality.to(DEVICE)
                val_loss = criterion(val_out, val_target)
                val_probs_np = torch.sigmoid(val_out).cpu().numpy().flatten() # Spłaszczenie
                val_target_cpu_np = val_target.cpu().numpy().flatten() # Spłaszczenie
                val_metric = roc_auc_score(val_target_cpu_np, val_probs_np)
                metric_name = "AUROC"

                if wandb.run is not None:
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
                        wandb.log({f"{task_name}/roc_curve_image": wandb.Image(roc_image)}, step=epoch)
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
                                ax_cm.text(j, i, str(cm[i, j]), va='center', ha='center', color='black' if cm[i,j] < cm.max()/2 else 'white')

                        buf_cm = io.BytesIO()
                        plt.savefig(buf_cm, format='png')
                        buf_cm.seek(0)
                        cm_image = Image.open(buf_cm)
                        wandb.log({f"{task_name}/confusion_matrix_image": wandb.Image(cm_image)}, step=epoch)
                        plt.close(fig_cm)
                        buf_cm.close()

                    except Exception as e:
                        print(f"Error logging wandb plots for Mortality: {e}")

            elif task_name == "LoS":
                val_target_log = torch.log1p(val_data.y_los.to(DEVICE))
                val_loss = criterion(val_out, val_target_log)
                val_preds_original_scale_np = torch.expm1(val_out).cpu().numpy().clip(min=0).flatten()
                val_target_original_scale_np = val_data.y_los.cpu().numpy().flatten()
                val_metric = np.sqrt(mean_squared_error(val_target_original_scale_np, val_preds_original_scale_np))
                metric_name = "RMSE"

                if wandb.run is not None:
                    try:
                        # Konwersja na listy Pythona dla tabeli scatter
                        data_scatter_list = [[float(true_val), float(pred_val)] for true_val, pred_val in zip(val_target_original_scale_np, val_preds_original_scale_np)]
                        table_scatter = wandb.Table(data=data_scatter_list, columns=["Actual LoS", "Predicted LoS"])
                        # Wykres scatter powinien działać bez problemu, więc go nie zmieniam
                        wandb.log({f"{task_name}/predictions_scatter": wandb.plot.scatter(
                            table_scatter, "Actual LoS", "Predicted LoS", title="Actual vs. Predicted LoS"
                        )}, step=epoch)
                    except Exception as e:
                        print(f"Error logging wandb plots for LoS: {e}")

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val {metric_name}: {val_metric:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_duration:.2f}s")

        if wandb.run is not None:
            log_metrics_dict = {
                #f"{task_name}/epoch": epoch, # Już logowane jako step
                f"{task_name}/train_loss": loss.item(),
                f"{task_name}/val_loss": val_loss.item(),
                f"{task_name}/val_{metric_name.lower()}": val_metric,
                f"{task_name}/learning_rate": optimizer.param_groups[0]['lr'],
                f"{task_name}/epoch_duration_sec": epoch_duration,
            }
            wandb.log(log_metrics_dict, step=epoch) # Logowanie metryk numerycznych

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

    if wandb.run is not None:
        wandb.save(model_path)
        wandb.summary[f"best_val_{metric_name.lower()}_{task_name.lower()}"] = best_val_metric
    return model, best_val_metric # Zwracamy również najlepszą metrykę

# --- Main Training Execution ---
def main(run_config_from_sweep=None):
    """
    Główna funkcja trenująca modele.
    Args:
        run_config_from_sweep (dict, optional): Słownik konfiguracji z wandb sweep.
    """
    if run_config_from_sweep is None:
        # Użyj domyślnych wartości, jeśli nie ma configu (np. przy standardowym uruchomieniu)
        current_run_config = {
            "learning_rate": LEARNING_RATE, "dim_hidden": DIM_HIDDEN,
            "num_gps_layers": NUM_GPS_LAYERS, "num_attn_heads": NUM_ATTN_HEADS,
            "dropout_rate": DROPOUT_RATE, "lap_pe_k_dim": LAP_PE_K_DIM,
            "sign_pe_k_dim": SIGN_PE_K_DIM, "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS, "patience_early_stopping": PATIENCE_EARLY_STOPPING,
            "cosine_t_0": COSINE_T_0, "cosine_t_mult": COSINE_T_MULT,
            "warmup_epochs": WARMUP_EPOCHS,
        }
        wandb_mode = "online"
        # Prosta nazwa dla pojedynczego uruchomienia, wandb doda unikalne ID
        run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        current_run_config = run_config_from_sweep
        wandb_mode = "online" # W trybie sweep wandb jest zawsze online
        # Nazwa dla sweep runa, agent wandb może ją nadpisać
        run_name = f"sweep_run_{wandb.run.id if wandb.run else time.strftime('%Y%m%d-%H%M%S')}"

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=current_run_config, # Przekazujemy pełną konfigurację
        name=run_name,
        mode=wandb_mode
    )

    # Pobierz wartości z wandb.config (które jest kopią current_run_config)
    _LEARNING_RATE = wandb.config.learning_rate
    _DIM_HIDDEN = wandb.config.dim_hidden
    _NUM_GPS_LAYERS = wandb.config.num_gps_layers
    _NUM_ATTN_HEADS = wandb.config.num_attn_heads
    _DROPOUT_RATE = wandb.config.dropout_rate
    _LAP_PE_K_DIM = wandb.config.lap_pe_k_dim
    _SIGN_PE_K_DIM = wandb.config.sign_pe_k_dim
    _WEIGHT_DECAY = wandb.config.weight_decay
    _EPOCHS = wandb.config.epochs
    _PATIENCE_EARLY_STOPPING = wandb.config.patience_early_stopping
    _COSINE_T_0 = wandb.config.cosine_t_0
    _COSINE_T_MULT = wandb.config.cosine_t_mult
    _WARMUP_EPOCHS = wandb.config.warmup_epochs

    print("Starting main training script execution...")
    print(f"Running with config: {wandb.config}")

    print("Loading and preprocessing training data for dimension inference and training...")
    train_graph, _, preprocessor = load_and_preprocess_data(
        TRAIN_CSV, fit_preprocessor=True, target_cols=['outcomeType', 'lengthofStay'], k_neighbors=10
    )
    if train_graph is None:
        print("Failed to load training data. Exiting.")
        if wandb.run: wandb.finish(exit_code=1)
        exit()

    num_numerical_features_fitted = 0
    if 'num' in preprocessor.named_transformers_ and preprocessor.named_transformers_['num'] != 'drop':
        num_numerical_features_fitted = len(preprocessor.transformers_[0][2])

    num_onehot_features_fitted = 0
    if 'cat' in preprocessor.named_transformers_ and preprocessor.named_transformers_['cat'] != 'drop':
        cat_transformer_fitted = preprocessor.named_transformers_['cat']
        onehot_encoder_fitted = cat_transformer_fitted.named_steps['onehot']
        if hasattr(onehot_encoder_fitted, 'get_feature_names_out'):
            try: # Handle cases where categorical_features might be empty
                num_onehot_features_fitted = len(onehot_encoder_fitted.get_feature_names_out(preprocessor.transformers_[1][2]))
            except Exception:
                 num_onehot_features_fitted = 0 # If no cat features were passed to onehot
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
            print(f"WARNING: Configured LAP_PE_K_DIM ({_LAP_PE_K_DIM}) differs from actual LapPE dim in data ({actual_lap_pe_dim}). Using actual: {actual_lap_pe_dim} for model init.")
            # Model should use actual_lap_pe_dim; wandb.config logs the intended _LAP_PE_K_DIM
    else:
        print("No LapPE found in data. LapPE dim for model will be 0.")

    actual_sign_pe_dim = 0
    if hasattr(train_graph, 'sign_pe') and train_graph.sign_pe is not None:
        actual_sign_pe_dim = train_graph.sign_pe.shape[1]
        print(f"SignPE found in data with dimension: {actual_sign_pe_dim}")
        if actual_sign_pe_dim != _SIGN_PE_K_DIM:
             print(f"WARNING: Configured SIGN_PE_K_DIM ({_SIGN_PE_K_DIM}) differs from actual SignPE dim in data ({actual_sign_pe_dim}). Using actual: {actual_sign_pe_dim} for model init.")
    elif _SIGN_PE_K_DIM > 0:
        print(f"WARNING: SIGN_PE_K_DIM is {_SIGN_PE_K_DIM} but no SignPE found in data. SignPE dim for model will be 0.")

    print("Loading and preprocessing validation data...")
    val_graph, _ = load_and_preprocess_data(
        VAL_CSV, preprocessor=preprocessor, fit_preprocessor=False,
        target_cols=['outcomeType', 'lengthofStay'], k_neighbors=10
    )
    if val_graph is None:
        print("Failed to load validation data. Exiting.")
        if wandb.run: wandb.finish(exit_code=1)
        exit()

    # --- Train Mortality Model ---
    print("\nInitializing Mortality Predictor...")
    mortality_model = MortalityPredictor(
        dim_in=DIM_IN_FEATURES_FROM_DATA, dim_h=_DIM_HIDDEN, num_layers=_NUM_GPS_LAYERS,
        num_heads=_NUM_ATTN_HEADS, lap_pe_dim=actual_lap_pe_dim, # Use actual dim from data
        sign_pe_dim=actual_sign_pe_dim, dropout=_DROPOUT_RATE # Use actual dim from data
    ).to(DEVICE)

    if wandb.run:
        wandb.config.update({
            "mortality_model_architecture": str(mortality_model),
            "input_features_dim_from_data": DIM_IN_FEATURES_FROM_DATA,
            "actual_lap_pe_dim_from_data_for_model": actual_lap_pe_dim, # Log what model actually uses
            "actual_sign_pe_dim_from_data_for_model": actual_sign_pe_dim # Log what model actually uses
        }, allow_val_change=True)

    if train_graph.y_mortality is not None:
        num_positive = torch.sum(train_graph.y_mortality == 1)
        num_negative = torch.sum(train_graph.y_mortality == 0)
        pos_weight_value = num_negative / (num_positive + 1e-6)
        print(f"Mortality: Negatives={num_negative}, Positives={num_positive}, Pos_weight={pos_weight_value:.2f}")
        criterion_mortality = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=DEVICE))
    else:
        print("Warning: y_mortality not found in train_graph. Using unweighted BCEWithLogitsLoss.")
        criterion_mortality = nn.BCEWithLogitsLoss()

    optimizer_mortality = optim.AdamW(mortality_model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    scheduler_mortality = CosineAnnealingWarmRestarts(optimizer_mortality, T_0=_COSINE_T_0, T_mult=_COSINE_T_MULT)

    mortality_model, best_auroc_mortality = train_model(mortality_model, train_graph, val_graph, criterion_mortality,
                                                        optimizer_mortality, scheduler_mortality, "Mortality",
                                                        _EPOCHS, _PATIENCE_EARLY_STOPPING, MORTALITY_MODEL_PATH)
    if wandb.run:
        wandb.summary["final_best_auroc_mortality"] = best_auroc_mortality

    # --- Train LoS Model ---
    print("\nInitializing LoS Predictor...")
    los_model = LoSPredictor(
        dim_in=DIM_IN_FEATURES_FROM_DATA, dim_h=_DIM_HIDDEN, num_layers=_NUM_GPS_LAYERS,
        num_heads=_NUM_ATTN_HEADS, lap_pe_dim=actual_lap_pe_dim, # Use actual dim from data
        sign_pe_dim=actual_sign_pe_dim, dropout=_DROPOUT_RATE # Use actual dim from data
    ).to(DEVICE)

    if wandb.run:
        wandb.config.update({"los_model_architecture": str(los_model)}, allow_val_change=True)

    criterion_los = nn.MSELoss()
    optimizer_los = optim.AdamW(los_model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    scheduler_los = CosineAnnealingWarmRestarts(optimizer_los, T_0=_COSINE_T_0, T_mult=_COSINE_T_MULT)

    los_model, best_rmse_los = train_model(los_model, train_graph, val_graph, criterion_los,
                                           optimizer_los, scheduler_los, "LoS",
                                           _EPOCHS, _PATIENCE_EARLY_STOPPING, LOS_MODEL_PATH)
    if wandb.run:
        wandb.summary["final_best_rmse_los"] = best_rmse_los

    print("\nTraining finished. Models saved to:")
    print(f"Mortality Model: {MORTALITY_MODEL_PATH}")
    print(f"LoS Model: {LOS_MODEL_PATH}")

    # Obliczanie i logowanie GLscore
    if wandb.run is not None:
        # Normalizacja RMSE (prosta, można dostosować)
        # Chcemy, aby wyższy wynik był lepszy, więc dla RMSE (gdzie niższy jest lepszy),
        # możemy użyć 1 - znormalizowany_rmse.
        # Normalizacja RMSE do zakresu ~0-1: normalized_rmse = rmse / (C + rmse), gdzie C to stała, np. 1 lub średnie RMSE.
        # Lub prostsze: exp(-k * rmse)
        # Dla tej implementacji użyjemy: 1 / (1 + RMSE) jako "wynik" dla LoS, aby był w (0,1] i wyższy był lepszy.
        if best_rmse_los is not None and best_auroc_mortality is not None:
            los_score_component = 1 / (1 + best_rmse_los) # Wyższy jest lepszy, zakres (0,1) dla RMSE > 0
            gl_score = best_auroc_mortality + los_score_component

            wandb.summary["los_score_component"] = los_score_component
            wandb.summary["GLscore"] = gl_score
            print(f"GLscore calculated: {gl_score:.4f} (AUROC: {best_auroc_mortality:.4f}, LoS_component: {los_score_component:.4f} from RMSE: {best_rmse_los:.4f})")
        else:
            print("Could not calculate GLscore because one or both primary metrics are missing.")

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None, help='Wandb sweep ID to run an agent for.')
    parser.add_argument('--sweep_config', type=str, default='sweep.yaml', help='Path to the sweep configuration file.')
    parser.add_argument('--project', type=str, default=WANDB_PROJECT, help='Wandb project name.')
    parser.add_argument('--entity', type=str, default=WANDB_ENTITY, help='Wandb entity (user or team).')
    parser.add_argument('--count', type=int, default=None, help='Number of runs for the sweep agent.')
    args = parser.parse_args()

    if args.sweep_id:
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

        wandb.agent(sweep_id=args.sweep_id, function=main_for_sweep, count=args.count, project=args.project, entity=args.entity)
    else:
        print("Starting a single training run (not a sweep agent).")
        main() # Wywołanie z config=None, użyje domyślnych wartości
