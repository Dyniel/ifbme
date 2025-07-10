## Agent Instructions and Coding Conventions

This file provides guidance for AI agents working on this codebase.

### General Principles:
1.  **Modularity:** Strive for modular code. Functions and classes should be well-defined and have a single responsibility.
2.  **Readability:** Write clear, concise, and well-commented code. Follow PEP 8 guidelines for Python.
3.  **Configuration:** Use the `configs/` directory for storing hyperparameters, file paths, and other settings. Avoid hardcoding these values directly in scripts.
4.  **Testing:** Add unit tests for new functionalities in the `tests/` directory. Aim for good test coverage.
5.  **Dependencies:** List all external library dependencies in `requirements.txt`. Specify versions if compatibility is critical.
6.  **Docstrings:** All functions and classes should have informative docstrings explaining their purpose, arguments, and return values.

### Specific Module Guidance:

*   **`data_utils/`**:
    *   Ensure data loading and preprocessing steps are efficient and handle potential edge cases (e.g., missing values).
    *   Balancing techniques like RSMOTE-GAN should be implemented carefully, with clear parameters and options.
*   **`features/`**:
    *   Feature engineering functions should be pure functions where possible (i.e., no side effects).
    *   Document the logic behind each feature clearly.
*   **`models/`**:
    *   Model implementations should allow for easy hyperparameter tuning.
    *   Ensure model saving and loading mechanisms are robust.
    *   For PyTorch models, ensure they can be easily moved to GPU/CPU.
*   **`self_supervised_learning/`**:
    *   Clearly document the SSL methods and their expected inputs/outputs.
    *   Provide scripts or clear instructions for running pre-training.
*   **`explainability/`**:
    *   Ensure that explainability methods are correctly applied to the corresponding models.
    *   Output from explainability tools should be easy to interpret.
*   **`scripts/`**:
    *   Scripts should be runnable from the command line and accept arguments for key parameters (e.g., config file path, dataset path).
    *   Include clear logging throughout the scripts.

### Running the Project:
*   (To be filled in as the project develops - e.g., how to train, evaluate)

### Data Sources:
*   (To be filled in - e.g., details about MIMIC-III/IV, COVID-ICU, HF-ICU if access or specific preprocessing is relevant for the agent)

### Preferred Libraries:
*   **Deep Learning:** PyTorch
*   **Tabular Models:** LightGBM, XGBoost
*   **General ML:** Scikit-learn
*   **NLP:** Hugging Face Transformers
*   **Graph ML:** PyTorch Geometric (preferred) or DGL

If any instructions are unclear, please ask for clarification.Tool output for `create_file_with_block`:
