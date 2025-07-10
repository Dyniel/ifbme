import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import joblib # For saving/loading model

class XGBoostMetaLearner:
    """
    Wrapper for XGBoost model, intended as a meta-learner in an ensemble.
    Handles training, prediction, and saving/loading.
    """
    # Default depth=3 as per AUROC spec for meta-learner
    def __init__(self, params=None, depth=3):
        """
        Args:
            params (dict, optional): Custom XGBoost parameters.
            depth (int): Max depth of trees. Default: 3.
        """
        self.default_params = {
            'objective': 'multi:softprob', # For multiclass probabilities
            'eval_metric': 'mlogloss',    # Evaluation metric
            'max_depth': depth,
            'eta': 0.1,                   # Learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'nthread': -1,                # Use all available cores
            # 'tree_method': 'hist' # Often faster, can be 'gpu_hist' if GPU supported XGBoost
        }
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        self.model = None
        self.best_iteration = None


    def train(self, X_train, y_train, X_val=None, y_val=None,
              num_boost_round=500, early_stopping_rounds=30):
        """
        Trains the XGBoost model.

        Args:
            X_train (np.ndarray): Training features (e.g., predictions from base models).
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation features for early stopping.
            y_val (np.ndarray, optional): Validation labels for early stopping.
            num_boost_round (int): Number of boosting rounds.
            early_stopping_rounds (int): Activates early stopping.
        """
        num_classes = len(np.unique(y_train))
        if num_classes == 2:
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = 'logloss' # Or 'auc'
        else:
            self.params['objective'] = 'multi:softprob'
            self.params['eval_metric'] = 'mlogloss'
            self.params['num_class'] = num_classes

        print(f"XGBoost using objective: {self.params['objective']} and eval_metric: {self.params['eval_metric']}")

        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
            evals_result = {} # To store training progress
        else:
            evals = [(dtrain, 'train')]
            evals_result = {}

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100, # Print evaluation results every 100 rounds
            evals_result=evals_result
        )

        self.best_iteration = self.model.best_iteration
        # print("XGBoost training evaluation results:", evals_result)
        print(f"XGBoost model trained. Best iteration: {self.best_iteration}")

    def predict_proba(self, X_test):
        """
        Predicts class probabilities.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        dtest = xgb.DMatrix(X_test)
        iteration_limit = self.best_iteration if self.best_iteration is not None else 0
        return self.model.predict(dtest, iteration_range=(0, iteration_limit))

    def predict(self, X_test):
        """
        Predicts class labels.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        if self.params.get('objective') == 'binary:logistic':
            return (proba > 0.5).astype(int)
        else: # multiclass
            return np.argmax(proba, axis=1)

    def save_model(self, filepath):
        """Saves the trained XGBoost model."""
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        joblib.dump({'model': self.model, 'params': self.params, 'best_iteration': self.best_iteration}, filepath)
        # self.model.save_model(filepath) # Native XGBoost save (model only)
        print(f"XGBoost model (with params and best_iteration) saved to {filepath} using joblib.")

    def load_model(self, filepath):
        """Loads an XGBoost model from file."""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.params = loaded_data['params'] # Load params used during training
        self.best_iteration = loaded_data.get('best_iteration') # Get best_iteration
        # self.model = xgb.Booster() # Init empty Booster
        # self.model.load_model(filepath) # Native XGBoost load
        print(f"XGBoost model loaded from {filepath}. Best iteration: {self.best_iteration}")


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("--- XGBoostMetaLearner Example ---")

    # 1. Generate dummy classification data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_classes=3, random_state=42) # Multiclass
    # X_bin, y_bin = make_classification(n_samples=1000, n_features=10, n_informative=5,
    #                                    n_classes=2, random_state=42) # Binary

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")


    # 2. Initialize and train XGBoostMetaLearner
    # Using default depth=3 as per spec for meta-learner
    xgb_params = {
        'eta': 0.05, # Learning rate
        # 'eval_metric': ['mlogloss', 'merror'] # Can specify multiple
    }
    xgb_wrapper = XGBoostMetaLearner(params=xgb_params, depth=3)

    print("\nTraining XGBoost model...")
    xgb_wrapper.train(
        X_train, y_train,
        X_val, y_val,
        num_boost_round=200, # Max rounds
        early_stopping_rounds=20
    )

    # 3. Make predictions
    print("\nMaking predictions...")
    y_pred_proba = xgb_wrapper.predict_proba(X_test)
    y_pred_labels = xgb_wrapper.predict(X_test)

    print(f"Predicted probabilities shape: {y_pred_proba.shape}")
    print(f"Predicted labels shape: {y_pred_labels.shape}")

    # 4. Evaluate (example)
    accuracy = accuracy_score(y_test, y_pred_labels)
    try:
        if len(np.unique(y_test)) > 2: # Multiclass
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        else: # Binary
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC (weighted OvR or binary): {auc_score:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC: {e}")


    # 5. Save and Load model
    model_path = "temp_xgb_meta_model.joblib"
    print(f"\nSaving model to {model_path}...")
    xgb_wrapper.save_model(model_path)

    print(f"Loading model from {model_path}...")
    loaded_xgb_wrapper = XGBoostMetaLearner(depth=3) # Init with same structural params
    loaded_xgb_wrapper.load_model(model_path)

    # Verify loaded model by predicting again
    y_pred_labels_loaded = loaded_xgb_wrapper.predict(X_test)
    accuracy_loaded = accuracy_score(y_test, y_pred_labels_loaded)
    print(f"Loaded Model Test Accuracy: {accuracy_loaded:.4f}")
    assert np.array_equal(y_pred_labels, y_pred_labels_loaded), "Predictions from saved and loaded model do not match."
    print("Predictions from saved and loaded model verified.")

    # Clean up
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}.")

    print("\n--- XGBoostMetaLearner Example Finished ---")

```
