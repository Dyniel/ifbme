import xgboost as xgb
import numpy as np
import joblib # For saving/loading model
import logging # For logging
import optuna # Import Optuna
from sklearn.metrics import roc_auc_score, log_loss # For Optuna objective

# It's good practice to have a logger instance per module.
logger = logging.getLogger(__name__)
# BasicConfig for logging can be set up in the main script that uses this class.
# If this class is run standalone (e.g. __main__), then basicConfig might be useful here.

class XGBoostMetaLearner:
    """
    A wrapper for the XGBoost model, specifically tailored for use as a meta-learner
    in an ensemble stacking setup. It handles model training, prediction,
    saving, and loading, while also managing XGBoost parameters and adapting
    to binary or multiclass classification tasks based on the training data.
    """
    def __init__(self, params=None, depth=3):
        """
        Initializes the XGBoostMetaLearner.

        The default parameters are chosen to be relatively conservative, suitable for
        a meta-learner that should not overfit to the OOF predictions of base models.
        The `depth` parameter is particularly important for this.

        Args:
            params (dict, optional): Custom XGBoost parameters. These will override or
                                     be added to the default parameters. Common parameters
                                     to tune include 'eta', 'subsample', 'colsample_bytree',
                                     'min_child_weight', 'gamma'.
            depth (int): Maximum depth of the trees. Default is 3. A shallow depth
                         is often recommended for meta-learners.

        Raises:
            TypeError: If `params` is provided but is not a dictionary.
        """
        self.default_params = {
            'objective': 'multi:softprob', # Default to multiclass probabilities; auto-adjusts in train()
            'eval_metric': 'mlogloss',    # Default eval metric for multiclass; auto-adjusts
            'max_depth': int(depth),      # Maximum tree depth
            'eta': 0.1,                   # Learning rate (alias: learning_rate)
            'subsample': 0.8,             # Subsample ratio of the training instances
            'colsample_bytree': 0.8,      # Subsample ratio of columns when constructing each tree
            'seed': 42,                   # Random seed for reproducibility
            'nthread': -1,                # Use all available CPU cores for training
            'scale_pos_weight': 2,        # Set scale_pos_weight to 2
            # Consider 'tree_method': 'hist' for potentially faster training on large datasets.
            # If GPU is available and XGBoost is compiled with GPU support, 'gpu_hist' can be used.
        }
        self.params = self.default_params.copy()
        if params:
            if not isinstance(params, dict):
                logger.error(f"Custom 'params' must be a dictionary, got {type(params)}.")
                raise TypeError(f"Custom 'params' must be a dictionary, got {type(params)}.")
            self.params.update(params) # Update defaults with any user-provided params

        self.model = None # Stores the trained XGBoost Booster object after calling train()
        self.best_iteration = None # Stores the best iteration if early stopping is used

        logger.debug(f"XGBoostMetaLearner initialized with effective params: {self.params}")

    def train(self, X_train, y_train, X_val=None, y_val=None,
              num_boost_round=500, early_stopping_rounds=30, verbose_eval=100,
              use_optuna=False, optuna_n_trials=20, optuna_timeout=None, wandb_run=None):
        """
        Trains the XGBoost model, with optional Optuna hyperparameter tuning.

        This method automatically adjusts 'objective' and 'eval_metric' parameters
        based on the number of unique classes detected in `y_train`, switching
        between binary and multiclass configurations. It supports early stopping
        if validation data (`X_val`, `y_val`) is provided.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation features for early stopping / Optuna.
            y_val (np.ndarray, optional): Validation labels for early stopping / Optuna.
            num_boost_round (int): Max boosting rounds (used if Optuna is off, or as max for Optuna trials).
            early_stopping_rounds (int, optional): For early stopping.
            verbose_eval (int or bool): Verbosity for XGBoost training.
            use_optuna (bool): If True, use Optuna for hyperparameter tuning.
            optuna_n_trials (int): Number of Optuna trials.
            optuna_timeout (int, optional): Timeout in seconds for Optuna study.
            wandb_run (wandb.sdk.wandb_run.Run, optional): Active W&B run object for logging Optuna results.

        Raises:
            ValueError: If X_train/y_train are None, shapes mismatch, or < 2 unique classes in y_train.
            TypeError: If X_train/y_train (or X_val/y_val if provided) are not numpy arrays.
            xgb.core.XGBoostError: If XGBoost training itself fails.
        """
        if X_train is None or y_train is None:
            logger.error("X_train and y_train cannot be None for training.")
            raise ValueError("X_train and y_train cannot be None.")
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            logger.error(f"X_train and y_train must be numpy arrays. Got types: {type(X_train)}, {type(y_train)}")
            raise TypeError("X_train and y_train must be numpy arrays.")
        if X_train.shape[0] != y_train.shape[0]:
            logger.error(f"Mismatch in number of samples: X_train has {X_train.shape[0]} and y_train has {y_train.shape[0]}.")
            raise ValueError("X_train and y_train must have the same number of samples.")

        try:
            num_unique_classes = len(np.unique(y_train))
        except Exception as e:
            logger.error(f"Could not determine number of unique classes from y_train: {e}")
            raise ValueError(f"Could not determine number of unique classes from y_train: {e}")

        if num_unique_classes < 2:
            logger.error(f"Number of unique classes in y_train must be at least 2. Found {num_unique_classes}.")
            raise ValueError(f"Number of unique classes in y_train must be at least 2. Found {num_unique_classes}.")

        self.num_classes = num_unique_classes # Store for Optuna objective

        # --- Auto-adjust objective and eval_metric based on number of classes ---
        # Store original params for Optuna, as self.params will be updated
        current_params = self.params.copy()
        user_defined_objective = current_params.get('objective', '')

        if self.num_classes == 2:
            if not user_defined_objective.startswith('binary:'):
                current_params['objective'] = 'binary:logistic'
                if 'eval_metric' not in current_params or current_params.get('eval_metric') == 'mlogloss':
                    current_params['eval_metric'] = ['logloss']
        else: # Multiclass
            if not user_defined_objective.startswith('multi:'):
                current_params['objective'] = 'multi:softprob'
                if 'eval_metric' not in current_params or current_params.get('eval_metric') == 'logloss':
                     current_params['eval_metric'] = ['mlogloss']
            current_params['num_class'] = self.num_classes

        if 'eval_metric' in current_params and isinstance(current_params['eval_metric'], str):
            current_params['eval_metric'] = [current_params['eval_metric']]

        # Update self.params with potentially auto-adjusted ones before Optuna or training
        self.params = current_params

        logger.info(f"XGBoostMetaLearner: Initial effective objective='{self.params['objective']}', "
                    f"eval_metric(s)='{self.params.get('eval_metric', 'None specified')}'")

        dtrain = xgb.DMatrix(X_train, label=y_train, nthread=self.params.get('nthread', -1))
        dval = None
        evals_list = [(dtrain, 'train')]
        use_early_stopping_for_final_train = False

        if X_val is not None and y_val is not None:
            if not isinstance(X_val, np.ndarray) or not isinstance(y_val, np.ndarray):
                raise TypeError("X_val and y_val must be numpy arrays if provided.")
            if X_val.shape[0] != y_val.shape[0]:
                 raise ValueError("X_val and y_val must have the same number of samples.")
            dval = xgb.DMatrix(X_val, label=y_val, nthread=self.params.get('nthread', -1))
            evals_list.append((dval, 'eval'))
            if early_stopping_rounds is not None and early_stopping_rounds > 0:
                use_early_stopping_for_final_train = True
        elif early_stopping_rounds is not None and early_stopping_rounds > 0:
            logger.warning("early_stopping_rounds specified but X_val/y_val not provided. Early stopping disabled.")

        if use_optuna:
            if dval is None:
                logger.warning("Optuna is enabled, but no validation set (X_val, y_val) was provided. "
                               "Optuna will optimize based on cross-validation within XGBoost if supported by params, "
                               "or this might lead to suboptimal tuning. Consider providing a validation set.")

            def xgb_objective(trial):
                xgb_params = {
                    'objective': self.params['objective'], # Use auto-detected objective
                    'eval_metric': self.params['eval_metric'], # Use auto-detected eval_metric
                    'eta': trial.suggest_float('eta', 1e-3, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2 reg
                    'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),   # L1 reg
                    'seed': self.params.get('seed', 42),
                    'nthread': self.params.get('nthread', -1),
                }
                if self.params.get('objective', '').startswith('multi:'):
                    xgb_params['num_class'] = self.num_classes

                trial_early_stopping_rounds = max(10, early_stopping_rounds // 2 if early_stopping_rounds else 10) # shorter for trials

                try:
                    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f"eval-{self.params['eval_metric'][0]}")

                    temp_model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=num_boost_round, # Max rounds for this trial
                        evals=evals_list,
                        early_stopping_rounds=trial_early_stopping_rounds,
                        verbose_eval=False, # Keep Optuna trials quiet
                        callbacks=[pruning_callback]
                    )

                    # Get the best score from the validation set
                    # XGBoost's train returns a Booster object. The score is in best_score if early stopping triggered.
                    # Or, we can predict on dval and calculate score.
                    if dval is not None:
                        preds_proba_val = temp_model.predict(dval, iteration_range=(0, temp_model.best_iteration if hasattr(temp_model, 'best_iteration') else 0))
                        if self.num_classes == 2:
                            score = roc_auc_score(y_val, preds_proba_val) # Maximize AUROC for binary
                        else: # Multiclass
                            score = log_loss(y_val, preds_proba_val) # Minimize logloss for multiclass
                        return score
                    else: # Fallback if no dval, rely on internal eval if any, or return a marker
                        # This case needs careful consideration: what metric to return if no validation set?
                        # Optuna usually requires a validation score.
                        logger.warning("Optuna trial running without external validation set. Pruning/evaluation might be suboptimal.")
                        # Return a value that Optuna can work with, e.g. last training loss if available
                        # For now, returning a neutral value or raising error might be better.
                        # Let's assume the pruning callback handles cases without dval based on train loss.
                        # The 'evals_result' from xgb.train could be parsed if needed.
                        # For simplicity, if no dval, we might have to skip proper Optuna evaluation here.
                        # This part of the objective function assumes dval is present for scoring.
                        return 0.0 # Should be improved if dval is not guaranteed for Optuna.


                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as e:
                    logger.warning(f"Optuna trial failed: {e}")
                    return float('inf') if self.params['eval_metric'][0] in ['logloss', 'mlogloss'] else 0.0 # Return worst score

            study_direction = 'maximize' if self.num_classes == 2 else 'minimize' # AUROC vs LogLoss
            study = optuna.create_study(direction=study_direction, sampler=optuna.samplers.TPESampler(seed=self.params.get('seed', 42)))
            study.optimize(xgb_objective, n_trials=optuna_n_trials, timeout=optuna_timeout)

            logger.info(f"Optuna study completed. Best trial: {study.best_trial.number}, Value: {study.best_value}")
            logger.info(f"Best parameters from Optuna: {study.best_params}")

            if wandb_run and hasattr(wandb_run, 'log'):
                wandb_run.log({
                    "meta_optuna_best_value": study.best_value,
                    "meta_optuna_best_params": study.best_params
                })

            # Update self.params with the best ones found by Optuna
            self.params.update(study.best_params)
            # Potentially adjust num_boost_round based on Optuna's findings if n_estimators was tuned,
            # or use the original num_boost_round for the final fit with best params.
            # For now, we'll use the original num_boost_round with early stopping for the final model.
            logger.info(f"Training final XGBoost Meta-Learner with Optuna's best_params: {self.params}")

        # --- Final Model Training (either with original params or Optuna-tuned params) ---
        evals_result_history = {}
        effective_early_stopping_rounds = early_stopping_rounds if use_early_stopping_for_final_train else None

        if not use_optuna:
            try:
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=evals_list,
                    early_stopping_rounds=effective_early_stopping_rounds,
                    verbose_eval=verbose_eval,
                    evals_result=evals_result_history
                )
            except xgb.core.XGBoostError as e:
                logger.error(f"XGBoost training failed: {e}")
                logger.debug(f"XGBoost parameters at time of failure: {self.params}")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during XGBoost training: {e}")
                raise

        if use_early_stopping_for_final_train and hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            self.best_iteration = self.model.best_iteration
            logger.info(f"XGBoostMetaLearner: Model training complete. Best iteration: {self.best_iteration} (due to early stopping).")
        else:
            self.best_iteration = self.model.num_boosted_rounds()
            logger.info(f"XGBoostMetaLearner: Model training complete. Total rounds: {self.best_iteration} (early stopping not triggered or disabled).")

        logger.debug(f"XGBoostMetaLearner: Training evaluation results history: {evals_result_history}")


    def predict_proba(self, X_test):
        """
        Predicts class probabilities for the input data.

        Args:
            X_test (np.ndarray): Test features, with the same number of columns as X_train.
                                 Must be a 2D numpy array.

        Returns:
            np.ndarray: Predicted class probabilities. Shape: (n_samples, n_classes).
                        For binary classification where `objective` is `binary:logistic`,
                        this method ensures a (n_samples, 2) output, representing
                        probabilities for class 0 and class 1 respectively.
        Raises:
            RuntimeError: If the model has not been trained.
            TypeError: If X_test is not a numpy array.
            ValueError: If X_test is not a 2D array.
            xgb.core.XGBoostError: If XGBoost prediction fails.
        """
        if self.model is None:
            logger.error("Cannot predict_proba: Model has not been trained yet. Call train() first.")
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        if not isinstance(X_test, np.ndarray):
            logger.error(f"X_test must be a numpy array for predict_proba. Got type: {type(X_test)}")
            raise TypeError("X_test must be a numpy array.")
        if X_test.ndim != 2:
            logger.error(f"X_test must be a 2D array for predict_proba. Got ndim: {X_test.ndim}")
            raise ValueError("X_test must be a 2D numpy array.")

        try:
            dtest = xgb.DMatrix(X_test, nthread=self.params.get('nthread', -1))
        except Exception as e:
            logger.error(f"Failed to create DMatrix for test data in predict_proba: {e}")
            raise

        iteration_limit = self.best_iteration if self.best_iteration is not None and self.best_iteration > 0 else 0

        try:
            raw_predictions = self.model.predict(dtest, iteration_range=(0, iteration_limit))
        except xgb.core.XGBoostError as e:
            logger.error(f"XGBoost prediction failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during XGBoost prediction: {e}")
            raise

        if self.params.get('objective') == 'binary:logistic':
            if raw_predictions.ndim == 1:
                prob_class1 = raw_predictions.reshape(-1, 1)
                prob_class0 = 1.0 - prob_class1
                return np.hstack([prob_class0, prob_class1])
            else:
                logger.warning(f"Unexpected output shape from predict() for binary:logistic: {raw_predictions.shape}. Attempting to infer.")
                if raw_predictions.shape[1] == 1:
                     prob_class1 = raw_predictions
                     prob_class0 = 1.0 - prob_class1
                     return np.hstack([prob_class0, prob_class1])
                elif raw_predictions.shape[1] == self.params.get('num_class', 2):
                    return raw_predictions
                else:
                    logger.error("Cannot reconcile predict() output for binary:logistic to (N,2) shape.")
                    raise ValueError("Inconsistent prediction output for binary:logistic objective.")

        elif self.params.get('objective', '').startswith('multi:softprob'):
            if raw_predictions.ndim == 2 and raw_predictions.shape[1] == self.params.get('num_class'):
                return raw_predictions
            else:
                logger.error(f"Unexpected output shape from predict() for multi:softprob: {raw_predictions.shape}. Expected num_class: {self.params.get('num_class')}")
                raise ValueError("Inconsistent prediction output for multi:softprob objective.")
        else:
            logger.warning(f"predict_proba called with an objective ('{self.params.get('objective')}') not explicitly handled for probability formatting. Returning raw predictions.")
            return raw_predictions

    def predict(self, X_test):
        """
        Predicts class labels for the input data.

        Args:
            X_test (np.ndarray): Test features. Must be a 2D numpy array.

        Returns:
            np.ndarray: Predicted class labels (integer-coded). Shape: (n_samples,).
        """
        probabilities = self.predict_proba(X_test)

        if probabilities is None or probabilities.shape[0] == 0:
            logger.warning("predict_proba returned empty or None, cannot determine class labels.")
            return np.array([])

        return np.argmax(probabilities, axis=1)

    def save_model(self, filepath):
        """
        Saves the trained XGBoost model, its parameters, and the best iteration number
        using joblib for Python object serialization.

        Args:
            filepath (str): Path to save the model file. Conventionally, this might
                            end with '.joblib' or '.pkl'.
        Raises:
            RuntimeError: If the model has not been trained.
            ValueError: If filepath is invalid.
            Exception: If joblib saving fails.
        """
        if self.model is None:
            logger.error("Cannot save model: No model has been trained. Call train() first.")
            raise RuntimeError("No model to save. Train the model first.")
        if not filepath or not isinstance(filepath, str):
            logger.error(f"Invalid filepath provided for saving model: '{filepath}'. Must be a non-empty string.")
            raise ValueError("Filepath must be a non-empty string.")

        try:
            model_data_to_save = {
                'model_booster': self.model,
                'params': self.params,
                'best_iteration': self.best_iteration
            }
            joblib.dump(model_data_to_save, filepath)
            logger.info(f"XGBoostMetaLearner: Model (booster, params, best_iteration) successfully saved to {filepath} using joblib.")
        except Exception as e:
            logger.error(f"XGBoostMetaLearner: Error saving model to {filepath} using joblib: {e}")
            raise

    def load_model(self, filepath):
        """
        Loads an XGBoost model, its parameters, and best iteration from a file
        previously saved by this class's `save_model` method (using joblib).

        Args:
            filepath (str): Path to the model file (e.g., a '.joblib' file).
        Raises:
            ValueError: If filepath is invalid or loaded file has incorrect structure.
            FileNotFoundError: If the model file does not exist.
            TypeError: If the loaded model booster is not of the expected type.
            Exception: If joblib loading fails for other reasons.
        """
        if not filepath or not isinstance(filepath, str):
            logger.error(f"Invalid filepath provided for loading model: '{filepath}'. Must be a non-empty string.")
            raise ValueError("Filepath must be a non-empty string.")

        try:
            loaded_model_data = joblib.load(filepath)

            if not isinstance(loaded_model_data, dict) or \
               'model_booster' not in loaded_model_data or \
               'params' not in loaded_model_data:
                logger.error("Loaded file does not contain the expected model data structure. Missing 'model_booster' or 'params'.")
                raise ValueError("Loaded file does not contain the expected model data structure.")

            self.model = loaded_model_data['model_booster']
            self.params = loaded_model_data['params']
            self.best_iteration = loaded_model_data.get('best_iteration')

            if not isinstance(self.model, xgb.Booster):
                 logger.error(f"Loaded 'model_booster' is not an xgb.Booster instance. Found type: {type(self.model)}")
                 raise TypeError(f"Loaded model_booster is not an xgb.Booster instance. Found type: {type(self.model)}")

            logger.info(f"XGBoostMetaLearner: Model successfully loaded from {filepath}.")
            logger.debug(f"Loaded model best_iteration: {self.best_iteration}, Loaded params: {self.params}")
        except FileNotFoundError:
            logger.error(f"XGBoostMetaLearner: Model file not found at {filepath}.")
            raise
        except Exception as e:
            logger.error(f"XGBoostMetaLearner: An error occurred loading model from {filepath}: {e}")
            raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import os

    logger.info("--- XGBoostMetaLearner Example ---")

    # --- Test Case 1: Binary Classification ---
    logger.info("\n--- Test Case 1: Binary Classification ---")
    X_bin, y_bin = make_classification(n_samples=500, n_features=10, n_informative=5,
                                       n_classes=2, random_state=42)
    X_train_bin, X_temp_bin, y_train_bin, y_temp_bin = train_test_split(X_bin, y_bin, test_size=0.4, random_state=42, stratify=y_bin)
    X_val_bin, X_test_bin, y_val_bin, y_test_bin = train_test_split(X_temp_bin, y_temp_bin, test_size=0.5, random_state=42, stratify=y_temp_bin)

    logger.info(f"Binary Train shapes: X={X_train_bin.shape}, y={y_train_bin.shape}")
    logger.info(f"Binary Val shapes: X={X_val_bin.shape}, y={y_val_bin.shape}")
    logger.info(f"Binary Test shapes: X={X_test_bin.shape}, y={y_test_bin.shape}")

    xgb_binary_wrapper = XGBoostMetaLearner(depth=3)
    logger.info("\nTraining binary XGBoost model...")
    xgb_binary_wrapper.train(
        X_train_bin, y_train_bin,
        X_val_bin, y_val_bin,
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=50
    )

    logger.info("\nMaking binary predictions...")
    y_pred_proba_bin = xgb_binary_wrapper.predict_proba(X_test_bin)
    y_pred_labels_bin = xgb_binary_wrapper.predict(X_test_bin)

    logger.info(f"Binary Predicted probabilities shape: {y_pred_proba_bin.shape}")
    assert y_pred_proba_bin.shape[0] == X_test_bin.shape[0], "Probabilities rows mismatch"
    assert y_pred_proba_bin.shape[1] == 2, "Binary predict_proba should return 2 columns."
    logger.info(f"Binary Predicted labels shape: {y_pred_labels_bin.shape}")
    assert y_pred_labels_bin.shape[0] == X_test_bin.shape[0], "Labels rows mismatch"

    accuracy_bin = accuracy_score(y_test_bin, y_pred_labels_bin)
    auc_score_bin = roc_auc_score(y_test_bin, y_pred_proba_bin[:, 1])
    logger.info(f"Binary Test Accuracy: {accuracy_bin:.4f}")
    logger.info(f"Binary Test AUC: {auc_score_bin:.4f}")

    # --- Test Case 2: Multiclass Classification ---
    logger.info("\n--- Test Case 2: Multiclass Classification ---")
    X_multi, y_multi = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                           n_classes=3, random_state=42)
    X_train_multi, X_temp_multi, y_train_multi, y_temp_multi = train_test_split(X_multi, y_multi, test_size=0.4, random_state=42, stratify=y_multi)
    X_val_multi, X_test_multi, y_val_multi, y_test_multi = train_test_split(X_temp_multi, y_temp_multi, test_size=0.5, random_state=42, stratify=y_temp_multi)

    logger.info(f"Multiclass Train shapes: X={X_train_multi.shape}, y={y_train_multi.shape}")
    logger.info(f"Multiclass Val shapes: X={X_val_multi.shape}, y={y_val_multi.shape}")
    logger.info(f"Multiclass Test shapes: X={X_test_multi.shape}, y={y_test_multi.shape}")
    logger.info(f"Number of classes in y_train_multi: {len(np.unique(y_train_multi))}")

    xgb_multi_wrapper = XGBoostMetaLearner(
        params={'eta': 0.05, 'eval_metric': ['mlogloss', 'merror']},
        depth=4
    )
    logger.info("\nTraining multiclass XGBoost model...")
    xgb_multi_wrapper.train(
        X_train_multi, y_train_multi,
        X_val_multi, y_val_multi,
        num_boost_round=200,
        early_stopping_rounds=20,
        verbose_eval=0
    )

    logger.info("\nMaking multiclass predictions...")
    y_pred_proba_multi = xgb_multi_wrapper.predict_proba(X_test_multi)
    y_pred_labels_multi = xgb_multi_wrapper.predict(X_test_multi)

    logger.info(f"Multiclass Predicted probabilities shape: {y_pred_proba_multi.shape}")
    num_expected_classes_multi = len(np.unique(y_multi))
    assert y_pred_proba_multi.shape[0] == X_test_multi.shape[0], "Probabilities rows mismatch multiclass"
    assert y_pred_proba_multi.shape[1] == num_expected_classes_multi, f"Multiclass predict_proba should return {num_expected_classes_multi} columns."
    logger.info(f"Multiclass Predicted labels shape: {y_pred_labels_multi.shape}")
    assert y_pred_labels_multi.shape[0] == X_test_multi.shape[0], "Labels rows mismatch multiclass"

    accuracy_multi = accuracy_score(y_test_multi, y_pred_labels_multi)
    try:
        auc_score_multi = roc_auc_score(y_test_multi, y_pred_proba_multi, multi_class='ovr', average='weighted')
        logger.info(f"Multiclass Test Accuracy: {accuracy_multi:.4f}")
        logger.info(f"Multiclass Test AUC (weighted OvR): {auc_score_multi:.4f}")
    except ValueError as e:
        logger.warning(f"Could not calculate Multiclass AUC: {e}")

    # --- Test Case 3: Save and Load model (using the multiclass model) ---
    model_path = "temp_xgb_meta_model.joblib"
    logger.info(f"\n--- Test Case 3: Save and Load ---")
    logger.info(f"Saving multiclass model to {model_path}...")
    try:
        xgb_multi_wrapper.save_model(model_path)

        logger.info(f"Loading model from {model_path}...")
        loaded_xgb_wrapper = XGBoostMetaLearner()
        loaded_xgb_wrapper.load_model(model_path)

        y_pred_labels_loaded = loaded_xgb_wrapper.predict(X_test_multi)
        accuracy_loaded = accuracy_score(y_test_multi, y_pred_labels_loaded)
        logger.info(f"Loaded Model Test Accuracy (Multiclass): {accuracy_loaded:.4f}")

        assert loaded_xgb_wrapper.params.get('eta') == 0.05, "Loaded params 'eta' does not match saved params."
        assert loaded_xgb_wrapper.params.get('max_depth') == 4, "Loaded params 'max_depth' does not match saved params."
        assert np.array_equal(y_pred_labels_multi, y_pred_labels_loaded), "Predictions from saved and loaded model do not match."
        logger.info("Predictions and parameters from saved and loaded model verified.")

    except Exception as e:
        logger.error(f"Error during save/load test: {e}")
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Cleaned up temporary model file: {model_path}.")

    logger.info("\n--- XGBoostMetaLearner Example Finished ---")
