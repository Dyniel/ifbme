import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import torch_geometric.utils as pyg_utils # Added for to_undirected

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os # For checking file existence

def load_and_preprocess_data(csv_path, preprocessor=None, fit_preprocessor=False, target_cols=None, k_neighbors=10, exclude_features=None):
    """
    Loads data from CSV, preprocesses features, and constructs a graph.

    Args:
        csv_path (str): Path to the CSV file.
        preprocessor (ColumnTransformer, optional): Fitted preprocessor.
                                                   If None and fit_preprocessor is True, a new one is created and fitted.
        fit_preprocessor (bool): If True, fit a new preprocessor or refit the provided one.
        target_cols (list, optional): List of target column names to extract.
        k_neighbors (int): Number of neighbors for k-NN graph.

    Returns:
        tuple: (torch_geometric.data.Data, pd.DataFrame or None, ColumnTransformer or None)
               - Graph data object
               - Target DataFrame (if target_cols provided)
               - Fitted preprocessor (if fit_preprocessor is True)
    """
    print(f"\n--- Starting data processing for: {csv_path} ---")
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        if fit_preprocessor:
            return None, None, None
        else:
            return None, None

    df = pd.read_csv(csv_path)
    print(f"Initial data shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")
    print(f"Data types before any processing:\n{df.dtypes}")

    # --- Date Feature Engineering ---
    date_cols = ['requestDate', 'admissionDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'admissionDate' in df.columns and 'requestDate' in df.columns:
        # Calculate stay_duration in days
        df['stay_duration_days'] = (df['admissionDate'] - df['requestDate']).dt.days
        # Fill potential NaNs if one date is missing or invalid, or if requestDate is after admissionDate
        df['stay_duration_days'] = df['stay_duration_days'].fillna(0).clip(lower=0)
        print("Created 'stay_duration_days'.")

    if 'admissionDate' in df.columns:
        df['admission_day_of_week'] = df['admissionDate'].dt.dayofweek
        df['admission_month'] = df['admissionDate'].dt.month
        df['admission_day_of_year'] = df['admissionDate'].dt.dayofyear
        # Fill NaNs for date components with a placeholder (e.g., -1 or median/mode if preferred)
        for col in ['admission_day_of_week', 'admission_month', 'admission_day_of_year']:
            df[col] = df[col].fillna(-1) # Using -1 for missing date parts
        print("Created 'admission_day_of_week', 'admission_month', 'admission_day_of_year'.")

    # --- Basic Cleaning & Initial Feature Engineering ---
    # 1. OutcomeType to numeric
    if 'outcomeType' in df.columns:
        print("Processing 'outcomeType'...")
        df['outcomeType'] = df['outcomeType'].replace({'Survival': 0, 'Death': 1})
        print(f"'outcomeType' unique values after mapping: {df['outcomeType'].unique()}")

    # 2. Handle 'Not applicable' and other string NaNs across likely numeric/categorical columns
    cols_to_check_na = ['glasgowScale', 'hematocrit', 'hemoglobin', 'leucocitos',
                        'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis', 'patientAge']
    print("Converting 'Not applicable' and similar to NaN for numeric columns...")
    for col in cols_to_check_na:
        if col in df.columns:
            original_na_count = df[col].isna().sum()
            df[col] = df[col].replace(['Not applicable', 'NA', 'NaN', 'nan', 'Not Applicable', 'Não aplicavel', 'Not informed'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_na_count = df[col].isna().sum()
            print(f"Column '{col}': {new_na_count - original_na_count} new NaNs from string replacement. Total NaNs: {new_na_count}")

    # 3. Blood Pressure: Parse 'blodPressure'
    if 'blodPressure' in df.columns:
        print("Parsing 'blodPressure'...")
        df['blodPressure_str'] = df['blodPressure'].astype(str).str.lower()
        bp_split = df['blodPressure_str'].str.split('x', expand=True)

        df['systolicPressure'] = pd.to_numeric(bp_split[0], errors='coerce')
        if bp_split.shape[1] > 1:
            # Clean common non-numeric parts like 'mmhg'
            df['diastolicPressure'] = pd.to_numeric(bp_split[1].str.replace(r'[^0-9\.]', '', regex=True), errors='coerce')
        else:
            df['diastolicPressure'] = np.nan

        df.drop(['blodPressure', 'blodPressure_str'], axis=1, inplace=True)
        print("Created 'systolicPressure' and 'diastolicPressure'. Dropped 'blodPressure'.")
        print(f"NaNs in systolicPressure: {df['systolicPressure'].isna().sum()}")
        print(f"NaNs in diastolicPressure: {df['diastolicPressure'].isna().sum()}")

    # 4. ICD Code: Extract first letter
    if 'icdCode' in df.columns:
        print("Processing 'icdCode' into 'icdCode_group'...")
        df['icdCode_group'] = df['icdCode'].astype(str).str[0].fillna('Unknown')
        df['icdCode_group'] = df['icdCode_group'].replace('nan', 'Unknown') # Handle 'nan' string
        print(f"'icdCode_group' unique values: {df['icdCode_group'].unique()}")

    # Drop original date columns and other specified columns
    cols_to_drop_initial = ['requestDate', 'admissionDate', 'icdCode']
    # requestBedType might not be in test data, handle conditionally
    if 'requestBedType' in df.columns and 'requestBedType' in cols_to_drop_initial:
        cols_to_drop_initial.append('requestBedType')

    df.drop(columns=[col for col in cols_to_drop_initial if col in df.columns], inplace=True, errors='ignore')
    print(f"Dropped initial columns. Remaining columns: {df.columns.tolist()}")

    # --- Define Feature Columns (after initial processing) ---
    temp_target_cols = target_cols if target_cols else []

    # Identify numerical features robustly
    # Define features based on typical expectations for this dataset
    # New date-derived features are numerical by default here, can be moved to categorical if needed (e.g. month as category)
    explicit_numerical_features = [
        'patientAge', 'glasgowScale', 'hematocrit', 'hemoglobin', 'leucocitos',
        'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
        'systolicPressure', 'diastolicPressure',
        'stay_duration_days', # New date feature
        'admission_day_of_week', 'admission_month', 'admission_day_of_year' # New date features
    ]

    explicit_categorical_features = [
        'requestType', 'admissionBedType', 'admissionHealthUnit',
        'patientGender', 'patientFfederalUnit', 'icdCode_group'
        # 'requestBedType' is often missing or inconsistent, handle its potential absence
    ]
    if 'requestBedType' in df.columns: # Add if column exists
        explicit_categorical_features.append('requestBedType')


    # Filter defined features to only those present in the current DataFrame
    numerical_features = [col for col in explicit_numerical_features if col in df.columns]
    categorical_features = [col for col in explicit_categorical_features if col in df.columns]

    # Ensure all columns are covered and handle 'Not applicable' more broadly
    all_defined_features = numerical_features + categorical_features
    for col in df.columns:
        if col not in all_defined_features and col not in temp_target_cols:
            # Convert 'Not applicable' etc. to NaN for all feature columns not yet processed by specific rules
            # This is a broader sweep than the initial cols_to_check_na
            if df[col].dtype == 'object': # only for object columns that might contain these strings
                df[col] = df[col].replace(['Not applicable', 'NA', 'NaN', 'nan', 'Not Applicable', 'Não aplicavel', 'Not informed'], np.nan)
                # Attempt conversion to numeric if it seems plausible after NA replacement
                # This helps catch numeric cols that were not in explicit_numerical_features but became numeric
                # if col not in categorical_features: # Don't try to convert explicitly categorical ones
                #    df[col] = pd.to_numeric(df[col], errors='ignore')


    # Refined identification:
    # Numerical: must be in numerical_features list AND have a numeric dtype after processing
    # Categorical: must be in categorical_features list OR have an object/category dtype
    final_numerical_features = []
    for col in numerical_features:
        # Try to convert to numeric one last time, in case some 'Not applicable' were missed or column was purely object before
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if pd.api.types.is_numeric_dtype(df[col]):
            final_numerical_features.append(col)
        else:
            print(f"Warning: Column {col} was in explicit_numerical_features but is not numeric. Moving to categorical or dropping if not in explicit_categorical_features.")
            if col not in categorical_features: # If not also defined as categorical, it might be problematic
                 # Decide: add to categorical, or log warning and it will be dropped by ColumnTransformer if not in either
                 categorical_features.append(col) # Safest: treat as categorical if numeric conversion failed

    final_categorical_features = []
    for col in categorical_features:
        if col in df.columns: # Ensure column still exists
            final_categorical_features.append(col)

    # Add any remaining object/category columns not explicitly listed but also not targets
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in final_categorical_features and col not in temp_target_cols and col not in final_numerical_features:
            print(f"Automatically adding column '{col}' of type {df[col].dtype} to categorical features.")
            final_categorical_features.append(col)

    # Remove duplicates that might have occurred if a col was in both lists or added multiple times
    final_numerical_features = sorted(list(set(final_numerical_features)))
    final_categorical_features = sorted(list(set(final_categorical_features)))

    # Ensure no overlap between final numerical and categorical lists
    common_cols = set(final_numerical_features).intersection(set(final_categorical_features))
    if common_cols:
        print(f"Warning: Columns {common_cols} are in both numerical and categorical lists. Removing from numerical.")
        final_numerical_features = [col for col in final_numerical_features if col not in common_cols]


    print(f"Final Numerical Features: {final_numerical_features}")
    print(f"Final Categorical Features: {final_categorical_features}")
    numerical_features = final_numerical_features
    categorical_features = final_categorical_features

    # --- Feature Exclusion ---
    if exclude_features:
        print(f"Excluding features based on exclude_features list: {exclude_features}")
        numerical_features = [col for col in numerical_features if col not in exclude_features]
        categorical_features = [col for col in categorical_features if col not in exclude_features]
        print(f"Numerical Features after exclusion: {numerical_features}")
        print(f"Categorical Features after exclusion: {categorical_features}")

    # Extract targets if specified
    if target_cols:
        targets_df = df[target_cols].copy()
        print(f"Targets extracted. Shape: {targets_df.shape}")
    else:
        targets_df = None
        print("No target columns specified.")

    feature_columns = numerical_features + categorical_features
    if not feature_columns:
        print("ERROR: No feature columns identified. Check data and feature definitions.")
        if fit_preprocessor:
            return None, None, None
        else:
            return None, None

    df_features = df[feature_columns].copy()
    print(f"Shape of DataFrame being sent to preprocessor: {df_features.shape}")


    # --- Preprocessing Pipelines ---
    if preprocessor is None:
        print("Preprocessor is None. Creating and fitting a new one.")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' # Drop any columns not specified
        )

        # Fit the preprocessor
        print(f"Fitting preprocessor on features: {feature_columns}")
        df_processed_np = preprocessor.fit_transform(df_features)
        print(f"Data processed. Shape after preprocessor fit_transform: {df_processed_np.shape}")

        try:
            onehot_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            all_feature_names = numerical_features + list(onehot_feature_names)
            print(f"Total features after one-hot encoding: {len(all_feature_names)}")
        except Exception as e:
            print(f"Could not get feature names out from one-hot encoder: {e}")
            all_feature_names = []


    elif fit_preprocessor: # Refit the provided preprocessor
        print("Refitting the provided preprocessor.")
        df_processed_np = preprocessor.fit_transform(df_features)
        print(f"Data processed. Shape after preprocessor fit_transform: {df_processed_np.shape}")
    else: # Use the already fitted preprocessor
        print("Using provided pre-fitted preprocessor.")
        df_processed_np = preprocessor.transform(df_features)
        print(f"Data processed. Shape after preprocessor transform: {df_processed_np.shape}")

    # --- Create PyTorch Geometric Data object ---
    x = torch.tensor(df_processed_np, dtype=torch.float)
    print(f"Node features tensor 'x' created with shape: {x.shape}")

    edge_index = knn_graph(x, k=k_neighbors, batch=None, loop=False)
    print(f"Edge index created with shape: {edge_index.shape}")

    graph_data = Data(x=x, edge_index=edge_index)

    # Add Laplacian Positional Encoding
    lap_pe_k = 8 # Default k for LapPE, can be parameterized in load_and_preprocess_data if needed
    # Check if graph_data has nodes and edges before applying transform
    if graph_data.num_nodes > 0 and graph_data.num_edges > 0 :
        print(f"Applying AddLaplacianEigenvectorPE with k={lap_pe_k}")
        # is_undirected needs to be True if the graph is meant to be undirected for PE calculation
        # PyG's knn_graph by default creates a directed graph (source to target for k-nearest).
        # For LapPE, an undirected graph's Laplacian is typically used.
        # We can make it undirected, or use is_undirected=False if the specific PE variant handles directed.
        # Let's make it undirected for standard LapPE.
        graph_data.edge_index = pyg_utils.to_undirected(graph_data.edge_index, num_nodes=graph_data.num_nodes)

        lap_pe_transform = AddLaplacianEigenvectorPE(k=lap_pe_k, attr_name='lap_pe', is_undirected=True)
        try:
            graph_data = lap_pe_transform(graph_data)
            print(f"Laplacian PE added. Shape: {graph_data.lap_pe.shape}")
        except Exception as e:
            print(f"Could not apply AddLaplacianEigenvectorPE: {e}. Num_nodes: {graph_data.num_nodes}, Num_edges: {graph_data.edge_index.size(1)}")
            # Fallback: add zero PE if transform fails
            graph_data.lap_pe = torch.zeros((graph_data.num_nodes, lap_pe_k), dtype=torch.float)
            print(f"Added zero LapPE as fallback. Shape: {graph_data.lap_pe.shape}")
    elif graph_data.num_nodes > 0 : # Nodes exist, but no edges from k-NN (e.g. k too small or isolated nodes)
        print(f"Skipping AddLaplacianEigenvectorPE as num_edges is 0. Adding zero LapPE.")
        graph_data.lap_pe = torch.zeros((graph_data.num_nodes, lap_pe_k), dtype=torch.float)
        print(f"Added zero LapPE. Shape: {graph_data.lap_pe.shape}")
    else: # No nodes
        print("Skipping AddLaplacianEigenvectorPE as there are no nodes in the graph.")


    if targets_df is not None:
        print("Adding target variables to graph_data object...")
        if 'outcomeType' in targets_df.columns:
            y_mortality = torch.tensor(targets_df['outcomeType'].values, dtype=torch.float) # BCEWithLogitsLoss expects float
            graph_data.y_mortality = y_mortality.unsqueeze(1) # Ensure shape [N, 1]
            print(f"y_mortality added. Shape: {graph_data.y_mortality.shape}")
        if 'lengthofStay' in targets_df.columns:
            y_los = torch.tensor(targets_df['lengthofStay'].values, dtype=torch.float)
            graph_data.y_los = y_los.unsqueeze(1) # Ensure shape [N, 1]
            print(f"y_los added. Shape: {graph_data.y_los.shape}")

    print(f"Final graph object: {graph_data}")
    print(f"--- Finished data processing for: {csv_path} ---\n")

    if fit_preprocessor:
        return graph_data, targets_df, preprocessor
    else:
        return graph_data, targets_df


if __name__ == '__main__':
    # Create dummy data files if they don't exist, for local testing.
    # This part is for local execution and won't run in the agent's sandbox.

    dummy_data_rows = 100
    base_columns = [
        'requestDate', 'requestType', 'requestBedType', 'admissionDate', 'admissionBedType',
        'admissionHealthUnit', 'patientGender', 'patientAge', 'patientFfederalUnit', 'icdCode',
        'blodPressure', 'glasgowScale', 'hematocrit', 'hemoglobin', 'leucocitos',
        'lymphocytes', 'urea', 'creatinine', 'platelets', 'diuresis',
        'outcomeType', 'lengthofStay'
    ]

    sample_data = {
        'requestDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, dummy_data_rows), unit='D'),
        'requestType': np.random.choice(['Adult', 'Pediatric', 'Geriatric'], dummy_data_rows),
        'requestBedType': np.random.choice(['UCI', 'Ward', 'Emergency'], dummy_data_rows),
        'admissionDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, dummy_data_rows), unit='D'),
        'admissionBedType': np.random.choice(['UCI', 'Ward', 'Semi-UCI'], dummy_data_rows),
        'admissionHealthUnit': np.random.choice(['Unit A', 'Unit B', 'Unit C', '2ª MOSSORÓ'], dummy_data_rows),
        'patientGender': np.random.choice(['M', 'F'], dummy_data_rows),
        'patientAge': np.random.randint(1, 100, dummy_data_rows),
        'patientFfederalUnit': np.random.choice(['RN', 'SP', 'MG', 'BA'], dummy_data_rows),
        'icdCode': [f"{chr(np.random.randint(65, 91))}{np.random.randint(0,10)}{np.random.randint(0,10)}.{np.random.randint(0,10)}" for _ in range(dummy_data_rows)],
        'blodPressure': [f"{np.random.randint(90, 180)}x{np.random.randint(50, 100)}" if np.random.rand() > 0.1 else f"{np.random.randint(90, 180)}X{np.random.randint(50, 100)}MMHG" for _ in range(dummy_data_rows)],
        'glasgowScale': [str(np.random.randint(3, 16)) if np.random.rand() > 0.1 else 'Not applicable' for _ in range(dummy_data_rows)],
        'hematocrit': np.random.uniform(20, 55, dummy_data_rows).round(1),
        'hemoglobin': np.random.uniform(7, 18, dummy_data_rows).round(1),
        'leucocitos': np.random.uniform(1, 30, dummy_data_rows).round(2) * 1000,
        'lymphocytes': np.random.randint(5, 70, dummy_data_rows),
        'urea': np.random.uniform(10, 200, dummy_data_rows).round(2),
        'creatinine': np.random.uniform(0.5, 10, dummy_data_rows).round(2),
        'platelets': np.random.randint(20, 600, dummy_data_rows),
        'diuresis': [str(np.random.randint(100, 3000)) if np.random.rand() > 0.05 else '9999' for _ in range(dummy_data_rows)],
        'outcomeType': np.random.choice(['Survival', 'Death'], dummy_data_rows, p=[0.8, 0.2]),
        'lengthofStay': np.random.randint(0, 100, dummy_data_rows)
    }
    sample_df = pd.DataFrame(sample_data)

    # Create dummy files if they don't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists('data/trainData.csv'):
        print("Creating dummy trainData.csv for local testing.")
        sample_df.sample(frac=0.7, random_state=42).to_csv('data/trainData.csv', index=False)

    if not os.path.exists('data/valData.csv'):
        print("Creating dummy valData.csv for local testing.")
        sample_df.sample(frac=0.15, random_state=1).to_csv('data/valData.csv', index=False)

    if not os.path.exists('data/testData.csv'):
        print("Creating dummy testData.csv for local testing.")
        dummy_test_df = sample_df.sample(frac=0.15, random_state=2).drop(columns=['outcomeType', 'lengthofStay'])
        dummy_test_df.to_csv('data/testData.csv', index=False)

    print("Starting example usage of load_and_preprocess_data...")

    # Example of features to exclude
    features_to_exclude_example = ['lymphocytes', 'admission_day_of_year']
    print(f"\n--- Example: Excluding features: {features_to_exclude_example} ---")


    print("\n--- Processing training data (fitting preprocessor) ---")
    train_graph_data, train_targets, fitted_preprocessor = load_and_preprocess_data(
        'data/trainData.csv',
        fit_preprocessor=True,
        target_cols=['outcomeType', 'lengthofStay'],
        k_neighbors=5, # Smaller k for smaller dummy data
        exclude_features=features_to_exclude_example
    )
    if train_graph_data:
        print(f"Train graph: {train_graph_data}")
        if train_targets is not None:
            print(f"Train targets head:\n{train_targets.head()}")
            print(f"Train y_mortality sample (from graph): {train_graph_data.y_mortality[:5] if hasattr(train_graph_data, 'y_mortality') else 'Not present'}")
            print(f"Train y_los sample (from graph): {train_graph_data.y_los[:5] if hasattr(train_graph_data, 'y_los') else 'Not present'}")
        if fitted_preprocessor:
            print("Preprocessor fitted successfully.")
    else:
        print("Failed to process training data.")

    if fitted_preprocessor:
        print("\n--- Processing validation data (using fitted preprocessor) ---")
        val_graph_data, val_targets = load_and_preprocess_data(
            'data/valData.csv',
            preprocessor=fitted_preprocessor,
            fit_preprocessor=False,
            target_cols=['outcomeType', 'lengthofStay'],
            k_neighbors=5
        )
        if val_graph_data:
            print(f"Validation graph: {val_graph_data}")
            if val_targets is not None:
                print(f"Validation targets head:\n{val_targets.head()}")
        else:
            print("Failed to process validation data.")

        if os.path.exists('data/testData.csv'):
            print("\n--- Processing test data (no targets, using fitted preprocessor) ---")
            test_graph_data, _ = load_and_preprocess_data(
                'data/testData.csv',
                preprocessor=fitted_preprocessor,
                fit_preprocessor=False,
                k_neighbors=5
            )
            if test_graph_data:
                print(f"Test graph: {test_graph_data}")
            else:
                print("Failed to process test data.")
        else:
            print("\n--- Skipping test data processing (testData.csv not found) ---")

        print("\n--- Verifying preprocessor content ---")
        num_feats_idx = 0
        cat_feats_idx = 1
        if not fitted_preprocessor.transformers_[0][2]: # if numerical_features was empty
            cat_feats_idx = 0

        print(f"Numerical features in preprocessor: {fitted_preprocessor.transformers_[num_feats_idx][2] if len(fitted_preprocessor.transformers_) > num_feats_idx and fitted_preprocessor.transformers_[num_feats_idx][0]=='num' else 'None or wrong index'}")
        if len(fitted_preprocessor.transformers_) > cat_feats_idx and fitted_preprocessor.transformers_[cat_feats_idx][0]=='cat':
             print(f"Categorical features in preprocessor: {fitted_preprocessor.transformers_[cat_feats_idx][2]}")
             try:
                onehot_cols = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(fitted_preprocessor.transformers_[cat_feats_idx][2])
                print(f"OneHotEncoder generated columns example: {list(onehot_cols[:5])} ... (total {len(onehot_cols)})")
             except Exception as e:
                print(f"Could not get feature names from OneHotEncoder: {e}")
        else:
            print("Categorical transformer not found or at unexpected index.")

    else:
        print("Preprocessor fitting failed, skipping dependent steps.")

    print("\nData processing utility script example usage finished.")
