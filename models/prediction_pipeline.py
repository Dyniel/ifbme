import joblib
import pandas as pd
import numpy as np
import os

class PredictionPipeline:
    def __init__(self, preprocessor, base_models, meta_learner, los_model, class_mapping, thresholds, config):
        """
        Initializes the PredictionPipeline.

        Args:
            preprocessor: The fitted preprocessing pipeline.
            base_models (dict): A dictionary of fitted base models (e.g., {'lgbm': model1, 'teco': model2}).
            meta_learner: The fitted meta-learner model.
            los_model: The fitted Length of Stay regression model.
            class_mapping (dict): Mapping from class names to integer labels.
            thresholds (dict): Dictionary of thresholds for prediction.
            config (dict): The configuration dictionary used during training.
        """
        self.preprocessor = preprocessor
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.los_model = los_model
        self.class_mapping = class_mapping
        self.thresholds = thresholds
        self.config = config
        self.death_label_value = self.class_mapping.get('Death')
        self.survival_label_value = self.class_mapping.get('Survival')


    def predict(self, X_raw_df: pd.DataFrame):
        """
        Makes predictions on new raw data.

        Args:
            X_raw_df (pd.DataFrame): The raw input data as a pandas DataFrame.

        Returns:
            tuple: A tuple containing two DataFrames:
                   - (pd.DataFrame): DTestimation predictions.
                   - (pd.DataFrame): LSestimation predictions.
        """
        if not isinstance(X_raw_df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        # --- 1. Preprocess Data ---
        X_processed = self.preprocessor.transform(X_raw_df)

        processed_feature_names = []
        try:
            processed_feature_names = self.preprocessor.get_feature_names_out()
        except Exception:
            num_processed_features = X_processed.shape[1]
            processed_feature_names = [f'proc_feat_{i}' for i in range(num_processed_features)]


        # --- 2. Base Model Predictions ---
        base_model_preds = {}
        # LGBM
        if 'lgbm' in self.base_models:
            lgbm_model = self.base_models['lgbm']
            base_model_preds['lgbm'] = lgbm_model.predict_proba(X_processed)

        # TECO (requires special data format)
        if 'teco' in self.base_models:
            # This part needs to be implemented based on how TECO model expects data
            # For now, let's assume it can take the processed numpy array and we handle it inside the model
            # This is a placeholder for the actual TECO prediction logic
            teco_model = self.base_models['teco']

            # We need to create a DataLoader for the new data
            from data_utils.sequence_loader import TabularSequenceDataset, basic_collate_fn
            from torch.utils.data import DataLoader

            df_teco = pd.DataFrame(X_processed, columns=processed_feature_names)
            teco_dataset = TabularSequenceDataset(
                data_frame=df_teco,
                targets=np.zeros(len(df_teco)), # Dummy targets
                feature_columns=processed_feature_names,
                target_column_name='dummy_target'
            )
            teco_loader = DataLoader(teco_dataset, batch_size=32, shuffle=False, collate_fn=basic_collate_fn)

            # This part needs to be implemented based on the TECO model's prediction logic
            # For now, we'll placeholder the prediction logic
            # base_model_preds['teco'] = teco_model.predict_proba(teco_loader)
            pass # Placeholder

        # --- 3. Meta-Learner Prediction ---
        # The order of concatenation must be the same as in training
        meta_features_list = []
        if 'lgbm' in base_model_preds:
            meta_features_list.append(base_model_preds['lgbm'])
        # if 'teco' in base_model_preds:
        #     meta_features_list.append(base_model_preds['teco'])

        X_meta = np.concatenate(meta_features_list, axis=1)

        final_probas_meta = self.meta_learner.predict_proba(X_meta)

        # --- 4. Apply Threshold for DTestimation ---
        # Using the average threshold from the folds for 'meta' model
        # A more robust approach would be to specify which threshold to use (e.g., from a specific fold or averaged)
        avg_meta_threshold = np.mean([v for k, v in self.thresholds.items() if 'meta' in k])

        death_class_idx = self.death_label_value
        predicted_outcome = np.where(final_probas_meta[:, death_class_idx] > avg_meta_threshold,
                                     self.death_label_value,
                                     self.survival_label_value)

        dt_estimation_df = pd.DataFrame({'predicted_outcome': predicted_outcome})


        # --- 5. Length of Stay Prediction ---
        predicted_los_transformed = self.los_model.predict(X_processed)
        predicted_los = np.expm1(predicted_los_transformed)
        predicted_los = np.maximum(0, predicted_los) # Ensure non-negativity

        ls_estimation_df = pd.DataFrame({'Estimated_LS': predicted_los})

        return dt_estimation_df, ls_estimation_df


    def save(self, filepath: str):
        """
        Saves the entire pipeline to a file.

        Args:
            filepath (str): The path to save the pipeline file.
        """
        try:
            joblib.dump(self, filepath)
            print(f"Pipeline saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving pipeline: {e}")
            raise

    @staticmethod
    def load(filepath: str):
        """
        Loads a pipeline from a file.

        Args:
            filepath (str): The path to the pipeline file.

        Returns:
            PredictionPipeline: The loaded pipeline instance.
        """
        try:
            pipeline = joblib.load(filepath)
            print(f"Pipeline loaded successfully from {filepath}")
            return pipeline
        except FileNotFoundError:
            print(f"Error: Pipeline file not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise
