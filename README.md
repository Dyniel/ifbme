# Clinical Prediction Pipeline (Development Stage)

This project aims to build a comprehensive pipeline for clinical prediction tasks. It incorporates modules for data balancing, feature engineering, self-supervised pre-training (conceptual), advanced modeling (including STM-GNN and TECO-Transformer - partially conceptual), ensembling, calibration, and explainability.

**Current Status:** The pipeline is in a **development stage**. Core components for tabular data processing, training LightGBM and XGBoost models within a Nested Cross-Validation (NCV) framework, and model calibration are functional and primarily use **dummy data by default**. Advanced deep learning models (STM-GNN, TECO-Transformer) and self-supervised learning modules are largely **conceptual or partially implemented** and also default to dummy data. Significant development is required to integrate real-world datasets and fully implement all proposed advanced features. This README details what is currently functional and what remains conceptual or requires further work.

The structure is designed based on specifications aimed at achieving high AUROC performance once fully developed.

## Project Structure

The project is organized as follows:

```
.
├── AGENTS.md                 # Instructions for AI agents
├── README.md                 # This file
├── requirements.txt          # Python package dependencies
├── sweep.yaml                # Example W&B sweep configuration (currently for dummy data)
├── configs/                  # Configuration files (YAML) for scripts
│   ├── __init__.py
│   └── dummy_train_config.yaml # Example config, generated/updated by scripts
├── data/                     # Directory for raw data (not version controlled by default)
│   ├── README.txt            # Instructions for data file placement
│   ├── example_script.py     # (Potentially an old example, not part of main pipeline)
│   ├── testData.csv          # Placeholder for test data
│   ├── trainData.csv         # Placeholder for training data
│   └── valData.csv           # Placeholder for validation data
├── data_utils/               # Utilities for data loading, preprocessing, balancing
│   ├── __init__.py
│   ├── balancing.py          # RSMOTEGAN implementation
│   ├── data_loader.py        # Script to load data from CSV files
│   ├── losses.py             # Custom loss functions (e.g., ClassBalancedFocalLoss - defined but not actively used)
│   └── preprocess.py         # Utility to create scikit-learn preprocessing pipelines
├── explainability/           # Tools for model calibration and explanation
│   ├── __init__.py
│   ├── calibration.py        # Isotonic Regression for model probability calibration (functional)
│   └── explain.py            # Placeholder for SHAP and Attention Rollout methods (conceptual)
├── features/                 # Modules for feature engineering
│   ├── __init__.py
│   ├── ontology_embeddings.py # Placeholder for ICD/ATC code embeddings (Node2Vec concept)
│   ├── text_embeddings.py    # Placeholder for using ClinicalBERT embeddings (conceptual)
│   └── trend_features.py     # Script to generate trend features (rolling mean, slope, variability - functional)
├── models/                   # Predictive models and neural network layers
│   ├── __init__.py
│   ├── lgbm_model.py         # Wrapper for LightGBM model (functional)
│   ├── main_model.py         # Example model integrating STMGNNLayer (development/example)
│   ├── meta_learner.py       # Wrapper for XGBoost meta-learner (functional)
│   ├── stm_gnn.py            # STM-GNN model and layers (highly conceptual, placeholder)
│   └── teco_transformer.py   # TECO-Transformer model (structure is defined, partially integrated)
├── notebooks/                # Jupyter notebooks for exploration and experimentation
│   └── .gitkeep
├── scripts/                  # Executable scripts for different pipeline stages
│   ├── __init__.py
│   ├── evaluate.py           # Script for evaluating trained models (conceptual/to be completed)
│   ├── predict.py            # Script for making predictions on new data (conceptual/to be completed)
│   ├── pretrain_ssl.py       # Script for self-supervised pre-training (conceptual, uses dummy data)
│   └── train.py              # Main script for NCV ensemble training (uses dummy data by default)
├── self_supervised_learning/ # Components for self-supervised learning
│   ├── __init__.py
│   ├── graphmae.py           # GraphMAE model (conceptual, depends on PyG)
│   └── mm_simclr.py          # MM-SimCLR model (conceptual)
└── tests/                    # Unit tests for various modules
    ├── __init__.py
    ├── test_balancing.py
    ├── test_lgbm_model.py
    └── test_trend_features.py
```

## Implemented Features

The following components are substantially implemented and considered functional, at least at a foundational level (some tested via `if __name__ == '__main__'` blocks or unit tests):

*   **Data Handling & Preprocessing:**
    *   Loading tabular data from CSV files (`data_utils/data_loader.py`).
    *   Creation of scikit-learn preprocessing pipelines for imputation, scaling, and one-hot encoding (`data_utils/preprocess.py`).
    *   Generation of trend features (rolling mean, slope, variability) from time-series data (`features/trend_features.py`).
    *   RSMOTE-GAN for data balancing (`data_utils/balancing.py`).
*   **Modeling (Tabular & Basic Sequence):**
    *   LightGBM model wrapper and training logic within the NCV framework (`models/lgbm_model.py`, `scripts/train.py`).
    *   XGBoost meta-learner wrapper and training logic (`models/meta_learner.py`, `scripts/train.py`).
    *   TECO-Transformer model structure (`models/teco_transformer.py`) is defined using standard PyTorch components. Basic training logic for it exists in `scripts/train.py` using a `TabularSequenceDataset`.
    *   Nested Cross-Validation (NCV) framework for ensemble training and evaluation (`scripts/train.py`).
*   **Model Enhancement & Utilities:**
    *   Model output probability calibration using Isotonic Regression (`explainability/calibration.py`).
    *   Configuration management via YAML files, including dynamic generation of dummy configurations (`configs/`, `scripts/`).
    *   Integration with Weights & Biases (W&B) for experiment tracking (`scripts/train.py`).
*   **Testing:**
    *   Unit tests for `balancing.py`, `lgbm_model.py`, and `trend_features.py` (`tests/`).

## Conceptual / Partially Implemented Features

Many components are currently conceptual, placeholders, or require significant further development to be fully operational for research or production use:

*   **Core Deep Learning Models:**
    *   **STM-GNN (`models/stm_gnn.py`):** The model structure is a high-level sketch. Critical components like temporal processing and global memory interaction are placeholders. The GNN layer implementation has basic fallbacks if `torch_geometric` is unavailable but is not a complete GNN layer without it. **Training of STM-GNN in `scripts/train.py` is non-functional and uses random predictions.** Requires `torch_geometric` for meaningful development.
    *   **TECO-Transformer (`models/teco_transformer.py`):** While the model structure using `torch.nn.TransformerEncoderLayer` is sound, its integration into `scripts/train.py` is basic. The `TabularSequenceDataset` is a simple wrapper and may not be suitable for all sequential EHR data. Full integration requires careful data preparation for sequential input and potentially more robust training loops.
*   **Self-Supervised Learning (SSL):**
    *   **GraphMAE (`self_supervised_learning/graphmae.py`):** A conceptual implementation of Graph Masked Autoencoders. It relies on `torch_geometric` (with basic fallbacks) for its GNN backbone. The pre-training script (`scripts/pretrain_ssl.py`) uses dummy graph data and a simplified training loop.
    *   **MM-SimCLR (`self_supervised_learning/mm_simclr.py`):** A conceptual framework for multimodal (e.g., structured EHR + text) contrastive learning. The pre-training script (`scripts/pretrain_ssl.py`) uses dummy encoders and dummy data.
*   **Data Pipelines for Advanced Models:**
    *   **General Data Loading in Scripts:** Both `scripts/train.py` (for TECO/STM-GNN if they were fully active) and `scripts/pretrain_ssl.py` currently operate using **dummy data generators**. Integrating real data loading and appropriate preprocessing for these advanced models is a major pending task.
    *   **Graph Data Pipeline (for STM-GNN, GraphMAE):** No specific pipeline exists yet for creating and processing graph snapshots from typical EHR data.
*   **Feature Engineering for Unstructured Data:**
    *   **Ontology Embeddings (`features/ontology_embeddings.py`):** Described as a "Node2Vec concept." No implementation is present.
    *   **Text Embeddings (`features/text_embeddings.py`):** Intended for using pre-trained ClinicalBERT embeddings. The module itself does not implement BERT or the embedding generation process.
*   **Explainability (Advanced):**
    *   **SHAP & Attention Rollout (`explainability/explain.py`):** These are listed as conceptual. SHAP could be integrated for tree models, but attention rollout would require specific implementations for Transformer/GNN models once they are functional.
*   **Core Scripts:**
    *   **`scripts/evaluate.py`:** Intended for evaluating trained models on a test set, including calibration and explainability. Currently conceptual and needs implementation.
    *   **`scripts/predict.py`:** Intended for making predictions on new, unseen data using a trained ensemble. Currently conceptual and needs implementation.
*   **Loss Functions:**
    *   **ClassBalancedFocalLoss (`data_utils/losses.py`):** While defined, it's noted in `scripts/train.py` as not being directly used by the NCV ensemble script at the top level. It's available for potential use in direct DL model training.

## What's Left to Implement / Future Work

To develop this project into a fully operational clinical prediction pipeline, the following areas require significant work:

1.  **Real Data Integration:**
    *   Modify `scripts/train.py` and `scripts/pretrain_ssl.py` to use the `data_utils.data_loader` and `data_utils.preprocess` modules for handling real-world datasets instead of the current dummy data generators.
    *   Develop robust data preprocessing pipelines tailored to specific EHR datasets (e.g., MIMIC-III/IV, eICU), including handling of missing values, feature scaling, and encoding specific to clinical data.
2.  **STM-GNN Development:**
    *   Complete the implementation of `models/stm_gnn.py`, particularly the `STMGNNLayer`'s temporal processing and global memory interaction components. This requires a good understanding of `torch_geometric`.
    *   Create a data pipeline to transform EHR data into graph snapshots suitable for STM-GNN.
    *   Replace the placeholder STM-GNN training logic in `scripts/train.py` with an actual training and validation loop.
3.  **Self-Supervised Learning (SSL) Enhancement:**
    *   For `GraphMAE` and `MM-SimCLR`, implement data loaders for real graph/multimodal data in `scripts/pretrain_ssl.py`.
    *   Refine the encoder architectures (e.g., GNN backbone for GraphMAE, structured/text encoders for MM-SimCLR) beyond the current dummy/conceptual versions.
    *   Establish a clear workflow for utilizing the pre-trained weights from SSL models in downstream supervised tasks.
4.  **Feature Engineering Implementation:**
    *   **Ontology Embeddings:** Implement a method (e.g., using Node2Vec or other graph embedding techniques) to generate embeddings from medical ontologies (like ICD, ATC codes if available with the dataset). Integrate these as features.
    *   **Text Embeddings:** Set up a pipeline to use pre-trained language models (e.g., ClinicalBERT via Hugging Face Transformers) to extract embeddings from clinical notes. Integrate these embeddings into the multimodal models or as features for tabular models.
    *   Integrate the functional `features/trend_features.py` module into the main data preprocessing pipeline in `scripts/train.py`.
5.  **TECO-Transformer Full Integration:**
    *   Develop a more specialized data loading and batching strategy for sequential EHR data to be used with `models/teco_transformer.py` within `scripts/train.py`. The current `TabularSequenceDataset` is a basic starting point.
    *   Conduct thorough hyperparameter tuning and validation for TECO as a base model.
6.  **Core Script Completion:**
    *   Implement `scripts/evaluate.py`: This script should load a trained ensemble (or individual models), evaluate it on a held-out test set, apply calibration, and generate explainability reports (e.g., SHAP values).
    *   Implement `scripts/predict.py`: This script should load a trained ensemble and allow making predictions on new, unlabeled data.
7.  **Advanced Explainability Methods:**
    *   Integrate SHAP for tree-based models (LightGBM, XGBoost meta-learner) within `scripts/evaluate.py`.
    *   For functional DL models (TECO, and eventually STM-GNN), implement appropriate explainability techniques (e.g., attention visualization, gradient-based methods, or specialized GNN explainers).
8.  **Robustness and Scalability:**
    *   Enhance error handling, logging, and model checkpointing across all scripts.
    *   Optimize data loading and training loops for larger datasets and more complex models.
    *   Expand unit test coverage.

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

Pipeline scripts are driven by YAML configuration files. Dummy configurations are generated on the first run if not present. **It is crucial to review and edit these configurations before running with actual data.**

### Step 1: Configuration (Essential for All Scripts)

1.  When a script like `python scripts/train.py` or `python scripts/pretrain_ssl.py` is run for the first time without a specified `--config` file, it will attempt to create a dummy one (e.g., `configs/dummy_train_config.yaml`, `configs/dummy_ssl_pretrain_config.yaml`).
2.  **Open and edit this YAML file:**
    *   **Data Paths:** For real data usage, update `data_paths` to point to your actual datasets (see specific script instructions below).
    *   **Model Hyperparameters:** Adjust parameters for LightGBM, XGBoost, TECO, STM-GNN, SSL models, etc., as needed.
    *   **Feature Engineering:** Configure settings for trend features, preprocessing steps (numerical/categorical columns).
    *   **Balancing & Ensemble:** Set options for RSMOTE-GAN and ensemble model selection.
    *   **Save Paths:** Specify correct paths for saving trained models or outputs. The default dummy configs often use relative paths like `models_trained/`; ensure these directories exist or update paths.

    An example snippet from `configs/dummy_train_config.yaml` related to data:
    ```yaml
    # ... (other parameters) ...
    # --- Data Paths (YOU MUST UPDATE FOR REAL DATA) ---
    # data_paths:
    #   train: "trainData.csv" # Name of your training data file in data_dir
    #   val: "valData.csv"     # Name of your validation data file in data_dir
    #   # test: "testData.csv" # Name of your test data file
    #   target_column: "Mortality" # Name of the target variable column in your CSVs
    #   # data_dir: "data/" # Base directory for data files (can be overridden)

    # --- Dummy Data Generation (USED BY DEFAULT in scripts/train.py) ---
    # These settings are used if the script is NOT modified to load real data.
    dummy_data_train_samples: 500
    dummy_data_val_samples: 100
    dummy_data_features: 10
    dummy_data_classes: 2
    # ... (other dummy data settings) ...
    ```

### Step 2: Running Scripts

#### A. Ensemble Model Training (`scripts/train.py`)

This script performs Nested Cross-Validation (NCV) to train an ensemble of models (LightGBM, TECO-Transformer, conceptually STM-GNN) and an XGBoost meta-learner.

*   **Default Behavior (Dummy Data):**
    ```bash
    python scripts/train.py --config configs/your_train_config.yaml
    ```
    By default, this script uses **internally generated dummy data** as specified in the `dummy_data_*` sections of the config. It will train LightGBM, a basic version of TECO-Transformer, and use placeholder predictions for STM-GNN.

*   **Running with Real CSV Data (Requires Code Modification):**
    1.  **Prepare Data:** Place your `trainData.csv` and `valData.csv` (or as named in your config) in the `data/` directory (or the directory specified by `data_dir` in your config). Ensure these CSVs contain a header row and your target column.
    2.  **Configure `your_train_config.yaml`:**
        *   Set `data_paths` (e.g., `train`, `val`, `target_column`).
        *   Define `preprocessing` settings: `numerical_cols`, `categorical_cols` (list column names from your CSVs), `imputation_strategy`, etc.
        *   Configure `trend_feature_params` if you want to generate trend features (provide `id_col`, `time_col`, `value_cols_trends`).
    3.  **Modify `scripts/train.py`:**
        *   **Crucial Step:** You **must** manually edit `scripts/train.py`. Locate the section responsible for data loading (search for "Using dummy data generation for this run.").
        *   Comment out or remove the dummy data generation block.
        *   Uncomment and adapt the conceptual real data loading block which uses `load_raw_data` from `data_utils.data_loader` and the `preprocessor` from `data_utils.preprocess`.
            ```python
            # Example modification in scripts/train.py:
            # logger.info("Using dummy data generation for this run.")
            # ... (dummy data generation code) ...
            # X_full_raw, y_full_raw = ...

            # INSTEAD, ENABLE REAL DATA LOADING:
            from data_utils.data_loader import load_raw_data
            from data_utils.preprocess import get_preprocessor # If using the standard preprocessor
            # from features.trend_features import make_trends # If generating trends

            logger.info("Attempting to load and preprocess real data...")
            # X_full_raw_df, y_full_raw_series = load_raw_data(config, base_data_path=config.get('data_paths',{}).get('data_dir',"data/"))

            # preprocessor = get_preprocessor(...) # Initialize based on config
            # X_full_processed = preprocessor.fit_transform(X_full_raw_df)
            # y_full_processed = y_full_raw_series.to_numpy()
            # ... (further processing like trend features, then assign to X_full_raw, y_full_raw for NCV) ...
            ```
            *(This is a conceptual guide; actual modifications will depend on your specific preprocessing chain.)*
    4.  Run the script:
        ```bash
        python scripts/train.py --config configs/your_train_config.yaml
        ```

#### B. Self-Supervised Pre-training (`scripts/pretrain_ssl.py` - Conceptual)

This script provides conceptual pre-training loops for GraphMAE and MM-SimCLR.

*   **Default Behavior (Dummy Data & Conceptual Models):**
    ```bash
    python scripts/pretrain_ssl.py --config configs/your_ssl_config.yaml
    ```
    This runs with **dummy data and simplified model interactions**. GraphMAE functionality also depends on `torch_geometric` being installed and working.

*   **For Meaningful Pre-training:**
    1.  **Data:** Prepare appropriate graph data (for GraphMAE) or paired multimodal data (for MM-SimCLR).
    2.  **Config:** Update `configs/your_ssl_config.yaml` with data paths and relevant SSL model parameters.
    3.  **Modify `scripts/pretrain_ssl.py`:** Replace dummy data generation with actual data loaders for your specific dataset format. You may also need to refine the encoder architectures within the script or the SSL model classes themselves.

#### C. Evaluating a Trained Model (`scripts/evaluate.py` - Conceptual)

This script is intended to evaluate a trained model/ensemble, apply calibration, and generate explanations.
```bash
# Conceptual command
python scripts/evaluate.py --config configs/your_train_config.yaml --model_path path/to/your/trained_model
```
**Status:** This script is largely a placeholder and needs to be implemented. It would load models trained by `scripts/train.py` and evaluate them on a test set (defined in config).

#### D. Making Predictions (`scripts/predict.py` - Conceptual)

This script is intended to load a trained model/ensemble and make predictions on new, unlabeled data.
```bash
# Conceptual command
python scripts/predict.py --config configs/your_train_config.yaml --model_path path/to/your/trained_model --input_data path/to/new_data.csv
```
**Status:** This script is also a placeholder and requires implementation.

### Step 3: Running Unit Tests

To run the existing unit tests (primarily for tabular data utilities and models):
```bash
python -m unittest discover tests
```
Or, for specific test files:
```bash
python -m unittest tests/test_balancing.py tests/test_trend_features.py tests/test_lgbm_model.py
```

## Important Considerations

*   **Dummy Data Default:** Scripts like `train.py` and `pretrain_ssl.py` **use internally generated dummy data by default**. Using real data requires manual code modification in these scripts to enable actual data loading and preprocessing functions.
*   **Conceptual Models:** STM-GNN, GraphMAE, and MM-SimCLR are **conceptual or partially implemented**. Their training in the current scripts is either placeholder (STM-GNN in `train.py`) or uses simplified loops with dummy data (`pretrain_ssl.py`). Significant development is needed to make them fully functional.
*   **`torch_geometric`:** For developing STM-GNN and GraphMAE, a working installation of PyTorch Geometric compatible with your PyTorch version is essential.
*   **Computational Resources:** Training advanced deep learning models and large ensembles can be computationally intensive. Ensure access to adequate hardware (e.g., GPUs for PyTorch models).
*   **Iterative Development:** This is a complex pipeline. Expect to debug and refine each component iteratively as you integrate actual data and fully implement the conceptual models.
*   **Model Persistence:** Scripts include conceptual paths for saving models (e.g., in `dummy_train_config.yaml`). Ensure these paths are correctly configured and that saving/loading mechanisms are robust for each model type once they are actively trained and saved.

This README provides a starting point for working with the pipeline. Good luck!
```
