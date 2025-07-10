# Clinical Prediction Pipeline

This project implements a comprehensive pipeline for clinical prediction tasks, incorporating advanced techniques for data balancing, feature engineering, self-supervised pre-training, state-of-the-art modeling (STM-GNN), ensembling, calibration, and explainability. The structure is designed based on specifications aimed at achieving high AUROC performance.

## Project Structure

```
.
├── AGENTS.md                 # Instructions for AI agents
├── README.md                 # This file
├── configs/                  # Configuration files (YAML)
│   └── (dummy configs will be generated here by scripts)
├── data_utils/               # Data loading, preprocessing, balancing, custom losses
│   ├── __init__.py
│   ├── balancing.py          # RSMOTEGAN (simplified SMOTE)
│   └── losses.py             # ClassBalancedFocalLoss
├── explainability/           # Calibration and model explanation tools
│   ├── __init__.py
│   ├── calibration.py        # Isotonic Regression calibrator
│   └── explain.py            # SHAP and Attention Rollout (conceptual)
├── features/                 # Feature engineering modules
│   ├── __init__.py
│   ├── ontology_embeddings.py # ICD/ATC code embeddings (Node2Vec concept)
│   ├── text_embeddings.py    # ClinicalBERT embeddings
│   └── trend_features.py     # Rolling mean, slope, variability
├── models/                   # Core predictive models and layers
│   ├── __init__.py
│   ├── lgbm_model.py         # LightGBM wrapper
│   ├── main_model.py         # Example model integrating STMGNNLayer
│   ├── meta_learner.py       # XGBoost meta-learner wrapper
│   ├── stm_gnn.py            # STM-GNN layer and model (conceptual)
│   └── teco_transformer.py   # TECO-Transformer model (conceptual)
├── notebooks/                # Jupyter notebooks for EDA and experimentation
│   └── .gitkeep
├── requirements.txt          # Python package dependencies
├── scripts/                  # Executable scripts for pipeline stages
│   ├── __init__.py
│   ├── evaluate.py           # Evaluation, calibration, explainability script
│   ├── predict.py            # Prediction script for the ensemble
│   ├── pretrain_ssl.py       # Self-supervised pre-training (conceptual)
│   └── train.py              # Main training script (ensemble learning)
├── ssl/                      # Self-supervised learning components
│   ├── __init__.py
│   ├── graphmae.py           # GraphMAE (conceptual)
│   └── mm_simclr.py          # MM-SimCLR (conceptual)
└── tests/                    # Unit tests
    ├── __init__.py
    ├── test_balancing.py
    ├── test_lgbm_model.py
    └── test_trend_features.py
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Python Virtual Environment:**
    (Recommended: Python 3.9+)
    ```bash
    python -m venv .venv
    ```
    Activate the environment:
    *   macOS/Linux: `source .venv/bin/activate`
    *   Windows: `.venv\\Scripts\\activate`

3.  **Install Dependencies:**
    The `requirements.txt` file lists necessary packages.
    ```bash
    pip install -r requirements.txt
    ```
    **Important Notes for PyTorch and PyTorch Geometric:**
    *   The `torch` and `torch-geometric` packages can have specific installation requirements based on your operating system, Python version, and whether you have a CUDA-enabled GPU.
    *   If `pip install -r requirements.txt` has issues with these, you may need to install them manually first, following instructions from their official websites:
        *   PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
        *   PyTorch Geometric: [Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) (ensure compatibility with your PyTorch version).

## Running the Pipeline

The pipeline scripts are driven by YAML configuration files located in the `configs/` directory. When scripts are run for the first time, they will generate dummy configuration files (e.g., `dummy_train_config.yaml`, `dummy_ssl_pretrain_config.yaml`) in this directory if they don't already exist.

**You MUST review and edit these generated configuration files before running the pipeline with your data.**

### Step 1: Configuration

1.  After running a script (e.g., `python scripts/train.py`) for the first time, a dummy config like `configs/dummy_train_config.yaml` will be created.
2.  Open this YAML file and modify it:
    *   Update placeholder data paths to point to your actual datasets.
    *   Adjust model hyperparameters, feature engineering settings, balancing options, and ensemble configurations as needed. The defaults are set based on the initial project specification aiming for high AUROC.
    *   Specify paths for saving trained models.

Here's an example structure of `configs/dummy_train_config.yaml` (the script will generate a more complete one):
```yaml
random_seed: 42
use_gpu: true
# Paths to your data -  YOU MUST UPDATE THESE or adapt data loading in scripts
# data:
#   train_path: "path/to/your/train_data.csv"
#   val_path: "path/to/your/validation_data.csv"
#   test_path: "path/to/your/test_data.csv"

# --- Dummy Data Generation (used if actual data loaders not integrated) ---
dummy_data_train_samples: 500
dummy_data_val_samples: 100   # Used as a proxy test set by some scripts
dummy_data_features: 10
dummy_data_classes: 2
dummy_data_train_weights: [0.9, 0.1]
dummy_data_val_weights: [0.8, 0.2]

balancing:
  use_rsmote_gan_in_cv: true
  rsmote_gan_params:
    k: 5
    minority_upsample_factor: 3.0

loss_function:
  type: ClassBalancedFocalLoss
  beta: 0.9999
  gamma: 2.0

ensemble:
  n_folds_for_oof: 5
  train_lgbm: true
  train_teco: true  # Note: TECO/STM-GNN training is conceptual
  train_stm_gnn: true
  train_meta_learner: true
  lgbm_params:
    num_leaves: 10000
    class_weight: 'balanced'
    # ... other lgbm params ...
    save_path: models_trained/lgbm_final.joblib
  teco_params:
    d_model: 512
    num_encoder_layers: 4
    # ... other teco params ...
    save_path: models_trained/teco_final.pth
  stm_gnn_params:
    layer_hidden_dim: 256
    num_gnn_layers: 5
    global_memory_dim: 128
    num_heads: 8
    # ... other stm_gnn params ...
    save_path: models_trained/stm_gnn_final.pth
  meta_learner_xgb_params:
    depth: 3
    # ... other xgb params ...
    save_path: models_trained/meta_learner_xgb.joblib
  soft_vote_weights:
    stm_gnn: 0.5
    lgbm: 0.3
    teco: 0.2

# evaluate_params will be added/used by evaluate.py
# evaluate_params:
#   model_to_evaluate: 'meta_learner'
#   apply_calibration: true
#   run_shap: true
```
*(Ensure the `models_trained` directory exists or adapt save paths)*

### Step 2: Running Scripts

*   **Self-Supervised Pre-training (Conceptual):**
    ```bash
    python scripts/pretrain_ssl.py --config configs/your_ssl_config.yaml
    ```
    (Edit `configs/dummy_ssl_pretrain_config.yaml` first, which is created on first run if `your_ssl_config.yaml` doesn't exist).

*   **Training the Ensemble Model:**
    ```bash
    python scripts/train.py --config configs/your_train_config.yaml
    ```
    This script handles the main ensemble training logic, including Out-Of-Fold (OOF) generation for the meta-learner.

*   **Evaluating a Trained Model:**
    ```bash
    python scripts/evaluate.py --config configs/your_train_config.yaml
    ```
    This script evaluates the model specified in the `evaluate_params` section of your config file. It performs calibration and conceptual explainability.

*   **Making Predictions on New Data:**
    ```bash
    python scripts/predict.py --config configs/your_train_config.yaml --data_path /path/to/your/new_data_for_prediction.csv
    ```
    (Update `--data_path` accordingly). This script loads the trained ensemble and predicts on new data.

### Step 3: Running Tests (Optional)

To run the placeholder unit tests:
```bash
python -m unittest tests/test_balancing.py tests/test_trend_features.py tests/test_lgbm_model.py
```
Or, to discover all tests in the `tests` directory:
```bash
python -m unittest discover tests
```

## Important Considerations

*   **Data Integration:** The current scripts primarily use **dummy data generators**. You will need to replace these with your actual data loading and preprocessing logic. This is a crucial step and involves modifying the data handling parts of the scripts (e.g., in `scripts/train.py`).
*   **Deep Learning Model Implementation:** The STM-GNN and TECO-Transformer models are complex. Their current implementations are **structural and conceptual placeholders**. To train and use them effectively, their internal architectures (especially `forward` methods and custom layers like `STMGNNLayer`) need to be fully implemented, debugged, and validated with real data.
*   **Computational Resources:** Training these models, particularly the deep learning components (STM-GNN, TECO-Transformer, SSL models) and large ensembles, will be computationally intensive. Ensure you have access to adequate hardware (e.g., GPUs for PyTorch models).
*   **Iterative Development:** This is a sophisticated pipeline. Expect to debug and refine each component iteratively as you integrate your actual data and fully implement the conceptual models.
*   **Model Persistence:** The scripts include conceptual paths for saving and loading models. Ensure these paths are correctly configured in your YAML files and that the saving/loading mechanisms are robust for each model type.

This README provides a starting point for working with the pipeline. Good luck!
```
