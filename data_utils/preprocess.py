import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_preprocessor(numerical_cols, categorical_cols, imputation_strategy='median', scale_numerics=True, handle_unknown_categorical='ignore'):
    """
    Creates a scikit-learn ColumnTransformer for general preprocessing.

    Args:
        numerical_cols (list): List of names of numerical columns.
        categorical_cols (list): List of names of categorical columns.
        imputation_strategy (str): Strategy for SimpleImputer (e.g., 'median', 'mean', 'most_frequent').
        scale_numerics (bool): Whether to apply StandardScaler to numerical features.
        handle_unknown_categorical (str): How OneHotEncoder should handle unknown categories.

    Returns:
        sklearn.compose.ColumnTransformer: A preprocessor object.
    """
    logger.info(f"Creating preprocessor: Numerical cols={numerical_cols}, Categorical cols={categorical_cols}")

    numerical_transformers = []
    if imputation_strategy:
        if imputation_strategy in ['median', 'mean']:
            numerical_transformers.append(('imputer', SimpleImputer(strategy=imputation_strategy, fill_value=0)))
        else:
            numerical_transformers.append(('imputer', SimpleImputer(strategy=imputation_strategy)))
    if scale_numerics:
        numerical_transformers.append(('scaler', StandardScaler()))

    numerical_pipeline = Pipeline(steps=numerical_transformers)

    categorical_transformers = []
    if imputation_strategy: # Can also impute categoricals, e.g. with most_frequent
         categorical_transformers.append(('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing'))) # Or 'most_frequent'
    categorical_transformers.append(('onehot', OneHotEncoder(handle_unknown=handle_unknown_categorical, sparse_output=False)))
    # Note: sparse_output=False for easier conversion to numpy/pandas if needed later.
    # For very high cardinality, sparse=True might be better with appropriate downstream handling.

    categorical_pipeline = Pipeline(steps=categorical_transformers)

    transformers_list = []
    if numerical_cols:
        transformers_list.append(('num', numerical_pipeline, numerical_cols))
    if categorical_cols:
        transformers_list.append(('cat', categorical_pipeline, categorical_cols))

    if not transformers_list:
        logger.warning("No numerical or categorical columns specified for preprocessing. Returning an 'empty' transformer.")
        # Return a "passthrough" transformer if no columns are specified, or handle as error
        return ColumnTransformer(transformers=[], remainder='passthrough')


    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')
    # remainder='passthrough' will keep columns not specified in numerical_cols or categorical_cols.
    # Use remainder='drop' to discard them.

    logger.info("Preprocessor created successfully.")
    return preprocessor

# Example conceptual usage (would be part of train.py or a data manager class)
# if __name__ == '__main__':
#     # Sample data
#     data = {
#         'age': [25, 30, np.nan, 35],
#         'salary': [50000, np.nan, 60000, 70000],
#         'city': ['New York', 'London', 'Paris', np.nan],
#         'gender': ['Male', 'Female', 'Male', 'Female'],
#         'target': [0, 1, 0, 1]
#     }
#     df = pd.DataFrame(data)
#     X = df.drop('target', axis=1)
#     y = df['target']

#     # Identify column types (this should be done more robustly in practice)
#     numerical_cols_example = ['age', 'salary']
#     categorical_cols_example = ['city', 'gender']

#     # Create preprocessor
#     # config_preprocessing = {'imputation': 'median', 'scaling': True, 'onehot_handle_unknown': 'ignore'} # from main config
#     preprocessor_instance = get_preprocessor(
#         numerical_cols=numerical_cols_example,
#         categorical_cols=categorical_cols_example,
#         # imputation_strategy=config_preprocessing['imputation'],
#         # scale_numerics=config_preprocessing['scaling'],
#         # handle_unknown_categorical=config_preprocessing['onehot_handle_unknown']
#     )

#     try:
#         # Fit on training data and transform
#         # In NCV, fit would happen on X_outer_train or X_inner_fold_train
#         X_transformed = preprocessor_instance.fit_transform(X)
#         logger.info("Data transformed successfully.")
#         logger.info(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")

#         # Get feature names after transformation (important for interpretability)
#         # This can be complex with ColumnTransformer, especially with 'passthrough'
#         # try:
#         #     feature_names_out = preprocessor_instance.get_feature_names_out()
#         #     logger.info(f"Transformed feature names: {feature_names_out}")
#         # except Exception as e:
#         #     logger.warning(f"Could not get feature names out directly: {e}")
#         #     # Fallback or manual construction might be needed depending on sklearn version and transformer complexity

#         # X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names_out if 'feature_names_out' in locals() else None)
#         # print(X_transformed_df.head())

#     except Exception as e:
#         logger.error(f"Error during preprocessing example: {e}")

def handle_missing_values_custom(df, strategy='median', columns=None):
    """(Alternative) Custom function for handling missing values."""
    logger.info(f"Handling missing values with strategy: {strategy}")
    # ... implementation ...
    return df

def encode_categorical_features_custom(df, columns=None, strategy='label'):
    """(Alternative) Custom function for encoding categorical features."""
    logger.info(f"Encoding categorical features with strategy: {strategy}")
    # ... implementation ...
    return df

def scale_numerical_features_custom(df, columns=None, scaler_type='standard'):
    """(Alternative) Custom function for scaling numerical features."""
    logger.info(f"Scaling numerical features with scaler: {scaler_type}")
    # ... implementation ...
    return df
