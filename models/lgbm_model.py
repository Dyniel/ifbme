import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import joblib # For saving/loading model

class LightGBMModel:
    """
    Wrapper for LightGBM model for classification.
    Handles training, prediction, and saving/loading.
    """
    # Defaults num_leaves=10000, class_weight='balanced' as per AUROC spec
    def __init__(self, params=None, num_leaves=10000, class_weight='balanced'):
        """
        Args:
            params (dict, optional): Custom LightGBM parameters.
                                     If None, uses default parameters with specified num_leaves and class_weight.
            num_leaves (int): Number of leaves for LightGBM. Default: 10000.
            class_weight (str or dict): Class weight parameter for LightGBM.
        """
        self.default_params = {
            'objective': 'multiclass', # Or 'binary' depending on the task
            'metric': 'multi_logloss', # Or 'binary_logloss', 'auc'
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': 0.05, # Typical starting point
            'feature_fraction': 0.9, # Typical starting point
            'bagging_fraction': 0.8, # Typical starting point
            'bagging_freq': 5,       # Typical starting point
            'class_weight': class_weight, # 'balanced' or a custom dict
            'verbose': -1, # Suppress LightGBM verbosity
            'n_jobs': -1,  # Use all available cores
            'seed': 42,
            # Add other relevant params from spec if needed
            # 'n_estimators': 1000, # Will be set by early stopping typically
        }
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None,
              num_boost_round=1000, early_stopping_rounds=50, categorical_feature='auto'):
        """
        Trains the LightGBM model.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features for early stopping.
            y_val (pd.Series or np.ndarray, optional): Validation labels for early stopping.
            num_boost_round (int): Number of boosting rounds.
            early_stopping_rounds (int): Activates early stopping.
                                         Stops training if validation metric doesn't improve.
            categorical_feature (str or list): Categorical features for LightGBM.
                                               'auto' or list of column names/indices.
            categorical_feature (str or list): Categorical features for LightGBM.
                                               'auto' or list of column names/indices.
        """
        # Use a local copy of params for modification within this method call
        current_params = self.params.copy()

        original_objective_from_init = current_params.get('objective', '').lower()
        original_metric_from_init = current_params.get('metric', '').lower()

        print(f"[DEBUG] LightGBMModel.train() - Start. Objective from __init__: {original_objective_from_init}, Metric from __init__: {original_metric_from_init}")

        is_regression_task = any(reg_obj in original_objective_from_init for reg_obj in
                                 ['regression', 'regression_l1', 'regression_l2', 'mae', 'mse', 'huber', 'quantile', 'poisson', 'gamma', 'tweedie'])

        if is_regression_task:
            current_params['objective'] = original_objective_from_init # Explicitly set regression objective
            # Ensure metric is regression-appropriate
            if not original_metric_from_init or not any(reg_metric in original_metric_from_init for reg_metric in ['mae', 'mse', 'rmse', 'huber', 'quantile', 'poisson', 'gamma', 'tweedie', 'l1', 'l2']): # l1/l2 can be metrics too
                current_params['metric'] = 'mae' # Default regression metric
            else:
                current_params['metric'] = original_metric_from_init # Keep user-defined regression metric

            if 'num_class' in current_params:
                del current_params['num_class']
        else:
            # Auto-detect for classification (binary/multiclass)
            num_unique_labels = len(np.unique(y_train))
            if num_unique_labels == 2:
                current_params['objective'] = 'binary'
                # Keep original metric if suitable (e.g. 'auc'), else default to 'binary_logloss'
                if not original_metric_from_init or \
                   any(cls_metric in original_metric_from_init for cls_metric in ['multi_logloss', 'multi_error', 'regression', 'mae', 'mse']): # if metric seems non-binary classification
                    current_params['metric'] = 'binary_logloss'
                else:
                    current_params['metric'] = original_metric_from_init

                if 'num_class' in current_params: # Not needed for binary
                    del current_params['num_class']
            else:
                current_params['objective'] = 'multiclass'
                current_params['num_class'] = num_unique_labels
                # Keep original metric if suitable, else default to 'multi_logloss'
                if not original_metric_from_init or \
                   any(cls_metric in original_metric_from_init for cls_metric in ['binary_logloss', 'auc', 'binary_error', 'regression', 'mae', 'mse']): # if metric seems non-multiclass classification
                     current_params['metric'] = 'multi_logloss'
                else:
                    current_params['metric'] = original_metric_from_init

        print(f"LightGBM using effective objective: {current_params.get('objective')} and effective metric: {current_params.get('metric')}")

        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature, free_raw_data=False)

        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=1))

        evals_result = {} # To store training progress
        callbacks.append(lgb.log_evaluation(period=100)) # Log every 100 rounds

        valid_sets = [lgb_train]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train,
                                  categorical_feature=categorical_feature, free_raw_data=False)
            valid_sets.append(lgb_val)
            valid_names.append('valid')

        self.model = lgb.train(
            current_params, # USE THE MODIFIED current_params
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            # evals_result=evals_result # This requires lgb.record_evaluation(evals_result) in callbacks
        )

        # print("LightGBM training evaluation results:", evals_result)
        print("LightGBM model trained.")

    def predict_proba(self, X_test):
        """
        Predicts class probabilities.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)

    def predict(self, X_test):
        """
        Predicts class labels.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        if self.params.get('objective') == 'binary':
            return (proba > 0.5).astype(int) # Threshold for binary
        else: # multiclass
            return np.argmax(proba, axis=1)

    def save_model(self, filepath):
        """Saves the trained LightGBM model."""
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        joblib.dump(self.model, filepath)
        # self.model.save_model(filepath) # Native LGBM save format
        print(f"LightGBM model saved to {filepath}")

    def load_model(self, filepath):
        """Loads a LightGBM model from file."""
        self.model = joblib.load(filepath)
        # self.model = lgb.Booster(model_file=filepath) # Native LGBM load format
        # Update params from loaded model if possible (joblib doesn't store them separately)
        # If using native format, some params are part of booster.
        print(f"LightGBM model loaded from {filepath}")
        # Infer objective and num_class from loaded model if possible (tricky with joblib)
        # This is a limitation of saving only the Booster object.
        # For now, assume params used at init are consistent with the loaded model's task.


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("--- LightGBMModel Example ---")

    # 1. Generate dummy classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_classes=3, random_state=42) # Multiclass
    # X_bin, y_bin = make_classification(n_samples=1000, n_features=20, n_informative=10,
    #                                    n_classes=2, random_state=42) # Binary

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # 2. Initialize and train LightGBMModel
    # Using smaller num_leaves for faster example
    lgbm_params = {
        'learning_rate': 0.1,
        'n_estimators': 200, # This is used by sklearn wrapper, but lgb.train uses num_boost_round
        'metric': ['multi_logloss', 'multi_error'] # Can specify multiple metrics
    }
    lgbm_wrapper = LightGBMModel(params=lgbm_params, num_leaves=31) # Default num_leaves is 10k

    print("\nTraining LightGBM model...")
    lgbm_wrapper.train(
        X_train, y_train,
        X_val, y_val,
        num_boost_round=200, # Max rounds
        early_stopping_rounds=20
    )

    # 3. Make predictions
    print("\nMaking predictions...")
    y_pred_proba = lgbm_wrapper.predict_proba(X_test)
    y_pred_labels = lgbm_wrapper.predict(X_test)

    print(f"Predicted probabilities shape: {y_pred_proba.shape}")
    print(f"Predicted labels shape: {y_pred_labels.shape}")

    # 4. Evaluate (example)
    accuracy = accuracy_score(y_test, y_pred_labels)
    # For multiclass AUC, need to handle one-vs-rest or one-vs-one
    try:
        if len(np.unique(y_test)) > 2: # Multiclass
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        else: # Binary
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1]) # Proba of positive class
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC (weighted OvR or binary): {auc_score:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC: {e}")


    # 5. Save and Load model
    model_path = "temp_lgbm_model.joblib"
    print(f"\nSaving model to {model_path}...")
    lgbm_wrapper.save_model(model_path)

    print(f"Loading model from {model_path}...")
    loaded_lgbm_wrapper = LightGBMModel(num_leaves=31) # Init with same structural params
    loaded_lgbm_wrapper.load_model(model_path)

    # Verify loaded model by predicting again
    y_pred_labels_loaded = loaded_lgbm_wrapper.predict(X_test)
    accuracy_loaded = accuracy_score(y_test, y_pred_labels_loaded)
    print(f"Loaded Model Test Accuracy: {accuracy_loaded:.4f}")
    assert np.array_equal(y_pred_labels, y_pred_labels_loaded), "Predictions from saved and loaded model do not match."
    print("Predictions from saved and loaded model verified.")

    # Clean up
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}.")

    print("\n--- LightGBMModel Example Finished ---")
