import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib
import os

from models.lgbm_model import LightGBMModel # Adjust import path

class TestLightGBMModel(unittest.TestCase):

    def setUp(self):
        # Generate data for binary and multiclass
        self.X_bin, self.y_bin = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        self.X_multi, self.y_multi = make_classification(n_samples=150, n_features=10, n_classes=3, n_informative=5, random_state=42) # 3 classes

        self.X_bin_train, self.X_bin_val, self.y_bin_train, self.y_bin_val = train_test_split(self.X_bin, self.y_bin, test_size=0.3, random_state=42)
        self.X_multi_train, self.X_multi_val, self.y_multi_train, self.y_multi_val = train_test_split(self.X_multi, self.y_multi, test_size=0.3, random_state=42)

        self.model_path = "temp_test_lgbm_model.joblib"

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_initialization(self):
        """Test model initialization with default and custom params."""
        model_default = LightGBMModel()
        self.assertEqual(model_default.params['num_leaves'], 10000)
        self.assertEqual(model_default.params['class_weight'], 'balanced')

        custom_params = {'learning_rate': 0.01, 'n_estimators': 50} # n_estimators not directly used by lgb.train
        model_custom = LightGBMModel(params=custom_params, num_leaves=50, class_weight=None)
        self.assertEqual(model_custom.params['num_leaves'], 50)
        self.assertIsNone(model_custom.params['class_weight'])
        self.assertEqual(model_custom.params['learning_rate'], 0.01)

    def test_train_predict_binary(self):
        """Test training and prediction for a binary classification task."""
        model = LightGBMModel(num_leaves=10, params={'n_estimators': 20}) # Small model for speed
        model.train(self.X_bin_train, self.y_bin_train, self.X_bin_val, self.y_bin_val,
                    num_boost_round=20, early_stopping_rounds=5)

        self.assertIsNotNone(model.model, "Model should be trained.")

        # Predict probabilities
        probas = model.predict_proba(self.X_bin_val)
        self.assertEqual(probas.shape, (len(self.X_bin_val), 2), "Probas shape incorrect for binary.")
        self.assertTrue(np.all((probas >= 0) & (probas <= 1)), "Probabilities out of [0,1] range.")
        self.assertTrue(np.allclose(np.sum(probas, axis=1), 1.0), "Probabilities do not sum to 1.")

        # Predict labels
        labels = model.predict(self.X_bin_val)
        self.assertEqual(labels.shape, (len(self.X_bin_val),), "Labels shape incorrect.")
        self.assertTrue(all(l in [0,1] for l in labels), "Labels not binary.")

    def test_train_predict_multiclass(self):
        """Test training and prediction for a multiclass classification task."""
        model = LightGBMModel(num_leaves=15, params={'n_estimators': 25}) # Small model
        model.train(self.X_multi_train, self.y_multi_train, self.X_multi_val, self.y_multi_val,
                    num_boost_round=25, early_stopping_rounds=5)

        self.assertIsNotNone(model.model)
        num_classes = len(np.unique(self.y_multi_train))
        self.assertEqual(model.params['num_class'], num_classes, "num_class not set correctly for multiclass.")

        probas = model.predict_proba(self.X_multi_val)
        self.assertEqual(probas.shape, (len(self.X_multi_val), num_classes), "Probas shape incorrect for multiclass.")
        self.assertTrue(np.all((probas >= 0) & (probas <= 1)))
        self.assertTrue(np.allclose(np.sum(probas, axis=1), 1.0))

        labels = model.predict(self.X_multi_val)
        self.assertEqual(labels.shape, (len(self.X_multi_val),))
        self.assertTrue(all(l in range(num_classes) for l in labels), "Labels not in expected multiclass range.")

    def test_save_load_model(self):
        """Test saving and loading the model."""
        model_orig = LightGBMModel(num_leaves=10)
        model_orig.train(self.X_bin_train, self.y_bin_train, num_boost_round=10)

        probas_orig = model_orig.predict_proba(self.X_bin_val)

        model_orig.save_model(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))

        model_loaded = LightGBMModel(num_leaves=10) # Init with same params for consistency
        model_loaded.load_model(self.model_path)
        self.assertIsNotNone(model_loaded.model)

        # Verify params (objective, num_class) might need to be re-inferred or stored if not using native save
        # For joblib, we are not storing self.params with the model, so this check is against initial params of loaded model
        # This is a known limitation of the current save/load with joblib if params change dynamically (like objective)
        # However, the core booster should be the same.

        probas_loaded = model_loaded.predict_proba(self.X_bin_val)
        self.assertTrue(np.allclose(probas_orig, probas_loaded), "Predictions from original and loaded model differ.")

    def test_predict_before_train(self):
        """Test that predict raises error if called before training."""
        model = LightGBMModel()
        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X_bin_val)
        with self.assertRaises(RuntimeError):
            model.predict(self.X_bin_val)

    def test_categorical_feature_handling(self):
        """Test training with categorical features (conceptual)."""
        # Create data with a categorical-like column (integer codes)
        X_cat_train = pd.DataFrame(self.X_bin_train, columns=[f'f{i}' for i in range(self.X_bin_train.shape[1])])
        X_cat_train['cat_feat'] = np.random.choice([0,1,2], size=len(X_cat_train))
        X_cat_val = pd.DataFrame(self.X_bin_val, columns=[f'f{i}' for i in range(self.X_bin_val.shape[1])])
        X_cat_val['cat_feat'] = np.random.choice([0,1,2], size=len(X_cat_val))

        model = LightGBMModel(num_leaves=10)
        # LightGBM can handle integer-coded categoricals if specified
        model.train(X_cat_train, self.y_bin_train, X_cat_val, self.y_bin_val,
                    categorical_feature=['cat_feat'], num_boost_round=10)
        self.assertIsNotNone(model.model)
        # Further checks could involve inspecting feature importances if possible or specific behavior.

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
