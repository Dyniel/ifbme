# data_utils/graph_constructor.py

"""
Contains functions to construct heterogeneous graph data objects for individual patients
using PyTorch Geometric.
"""
import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from data_utils.graph_schema import NODE_TYPES, EDGE_TYPES # Import schema

# Placeholder for global concept mappers.
# In a real implementation, these would be pre-computed from the entire dataset.
# Example:
# ICD_TO_ID = {'I10': 0, 'J44.9': 1, ...}
# VITAL_TO_ID = {'HR': 0, 'RR': 1, ...}
# MED_TO_ID = {'C09AA02': 0, ...}
# PROC_TO_ID = {'8-101': 0, ...}

def get_global_concept_mappers(full_dataset_df, vital_cols, icd_col, med_cols, proc_cols):
    """
    Scans the full dataset to create mappings from concept strings to unique integer IDs.
    This should be called once and the mappers reused.
    """
    mappers = {}

    # Vital/Lab concepts
    # Assuming vital_cols is a list of column names like ['HR', 'WBC', 'Lactate']
    # These are directly the names of the concepts.
    unique_vitals = sorted(list(set(vital_cols)))
    mappers['vital_to_id'] = {name: i for i, name in enumerate(unique_vitals)}

    # Diagnosis codes
    # Assuming icd_col contains codes, possibly multiple per entry separated by a delimiter
    # For simplicity, assuming they are pre-processed into a list of single codes if necessary
    # or that patient_df will have them in a suitable format.
    # This part would need refinement based on actual data format.
    # For now, let's assume we can extract unique codes.
    # Example: all_icds = full_dataset_df[icd_col].dropna().unique()
    # mappers['diagnosis_to_id'] = {code: i for i, code in enumerate(sorted(list(set(all_icds))))}
    # This is highly dependent on how ICD codes are stored and if they need splitting/cleaning.
    # For now, we'll assume it's passed or handled before build_patient_graph.
    # Placeholder - this needs to be robustly implemented based on actual data.

    # For now, we'll assume the calling code (e.g., Dataset __init__) handles mapper creation.
    # This function is more of a conceptual placeholder for where it *could* happen.

    # Let's create dummy mappers for now if specific columns are not fully defined yet.
    # In a real scenario, these would be populated by iterating over the *entire* dataset once.

    # Placeholder - these should be properly generated from the full dataset.
    # For the purpose of `build_patient_graph` structure, we'll assume they are passed in.
    # Example of how they might be built if we had column names:
    # if icd_col and icd_col in full_dataset_df:
    #     all_icds = set()
    #     for item_list in full_dataset_df[icd_col].dropna(): # Assuming item_list is a list of codes
    #         for code in item_list: all_icds.add(code)
    #     mappers['diagnosis_to_id'] = {code: i for i, code in enumerate(sorted(list(all_icds)))}
    # else:
    #     mappers['diagnosis_to_id'] = {}
    # Similar for med_cols, proc_cols if they are lists of columns or single columns with lists of codes.

    # This function's main purpose is to highlight the need for these mappers.
    # The actual creation will likely be in the Dataset's __init__ or a preprocessing script.
    print("Warning: get_global_concept_mappers is a placeholder. Mappers should be pre-computed.")
    return mappers


def build_patient_graph(patient_df, patient_id, target_variable_name, label_timestamp,
                        time_bin_size_hours,
                        vital_col_names, # e.g., ['HR', 'RR', 'Temp', 'Lactate']
                        diagnosis_col_name, # Name of column holding diagnosis codes (e.g., 'icdCode')
                        medication_col_name, # Name of column holding medication codes
                        procedure_col_name,  # Name of column holding procedure codes
                        global_concept_mappers,
                        max_graph_history_hours=None): # e.g., 24, to only use data up to 24h before label_timestamp
    """
    Constructs a heterogeneous graph for a single patient up to a specified snapshot time.

    Args:
        patient_df (pd.DataFrame): DataFrame for a single patient, sorted by time.
                                   Must contain a timestamp column (e.g., 'timestamp_col'),
                                   vital/lab columns, and columns for diagnosis, medication, procedure codes.
        patient_id (any): Unique identifier for the patient.
        target_variable_name (str): Name of the target variable column (e.g. 'outcomeType')
        label_timestamp (pd.Timestamp): The timestamp at which the label is defined (e.g., discharge time or death time).
                                         The graph will be constructed using data *before* this time.
        time_bin_size_hours (int): Size of time bins in hours.
        vital_col_names (list): List of column names for vital signs and lab results.
        diagnosis_col_name (str): Column name for diagnosis codes. Expected to contain lists of codes or single codes.
        medication_col_name (str): Column name for medication codes.
        procedure_col_name (str): Column name for procedure codes.
        global_concept_mappers (dict): Dict containing mappings like:
                                       {'vital_to_id': {...}, 'diagnosis_to_id': {...}, ...}
        max_graph_history_hours (int, optional): If provided, only include data from this many hours
                                                 before `label_timestamp`.

    Returns:
        torch_geometric.data.HeteroData: A heterogeneous graph for the patient.
                                         Includes `data.y` for the target label.
                                         Returns None if essential data is missing or graph cannot be formed.
    """
    data = HeteroData()

    # 0. Ensure patient_df has a proper timestamp column (e.g., 'timestamp_col', 'requestDate_unix')
    # --- Time Handling and Snapshotting ---
    # Ensure 'time_rel_hours' column (hours since admission) exists. This is expected to be pre-calculated by the Dataset.
    if 'time_rel_hours' not in patient_df.columns:
        print(f"ERROR: Patient {patient_id} DataFrame missing 'time_rel_hours' column. Cannot build graph.")
        return None

    # Determine the snapshot window based on label_timestamp and max_graph_history_hours.
    # This part assumes label_timestamp is an absolute timestamp and needs to be related to
    # the patient's admission time to correctly use 'time_rel_hours'.
    # For this, we'd also need the patient's absolute admission time.
    # Let's assume patient_df ALREADY contains only events within the desired snapshot window
    # AND that 'time_rel_hours' is correctly calculated and filtered.
    # This simplification means the Dataset's get() method is responsible for providing a pre-snapshotted patient_df.

    if patient_df.empty:
        # print(f"Warning: Patient {patient_id} has no data for the snapshot period. Creating empty graph structure.")
        # Create a valid empty HeteroData object for schema consistency
        for node_type in NODE_TYPES:
            data[node_type].x = torch.empty((0, 1)) # Dummy feature dim of 1
            data[node_type].num_nodes = 0
        # No edges to create. data.y will be set by the caller (Dataset)
        return data

    # --- 1. Create Time-slice (T_t) nodes ---
    # time_rel_hours should be 0 for the first event post-admission if filtered correctly.
    # min_time_rel = patient_df['time_rel_hours'].min() # Should be >= 0
    max_time_rel = patient_df['time_rel_hours'].max()

    # Determine the number of time bins needed to cover the duration of the snapshot
    # If max_time_rel is 23.5 hours and bin_size is 1, we need bins 0...23, so 24 bins.
    # Bin index = floor(time_rel_hours / time_bin_size_hours)
    num_time_bins = int(np.floor(max_time_rel / time_bin_size_hours)) + 1

    if num_time_bins <= 0: # Should not happen if patient_df is not empty
        print(f"Warning: Patient {patient_id} calculated num_time_bins <= 0 ({num_time_bins}). Max_time_rel: {max_time_rel}. Creating empty graph.")
        for node_type in NODE_TYPES:
            data[node_type].x = torch.empty((0, 1))
            data[node_type].num_nodes = 0
        return data

    data['timeslice'].num_nodes = num_time_bins

    # T_t node features:
    # Part 1: Sin/Cos positional encoding for time.
    # Let's define a time_embedding_dim, e.g., 16 or 32.
    time_embedding_dim = 16 # Example, can be configurable

    pe = torch.zeros(num_time_bins, time_embedding_dim)
    position = torch.arange(0, num_time_bins, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, time_embedding_dim, 2).float() * (-math.log(10000.0) / time_embedding_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:,:time_embedding_dim // 2]) # Ensure div_term matches if dim is odd for cos part

    # Part 2: Aggregated vitals/labs for T_t features.
    # Initialize with NaNs or zeros, then fill.
    # The order of features in this tensor should match vital_col_names.
    num_vital_features = len(vital_col_names)
    aggregated_vital_features = torch.full((num_time_bins, num_vital_features), float('nan'))

    # Combine time encoding with space for aggregated vitals
    data['timeslice'].x = torch.cat([pe, aggregated_vital_features], dim=1)
    # data['timeslice'].x will be updated in place later with actual aggregated values.

    # --- 2. Identify active V/D/M/P (concept) nodes and prepare for edges ---

    # Store mappings from local (patient-specific active) concept index to global concept ID
    # This is useful if we only want to create embedding lookups for active concepts in this graph.
    # Alternatively, the GNN model's embedding layer can map from global_ids directly.
    # For HeteroData, we typically define num_nodes for each type based on the max global_id + 1
    # if using global embeddings, or based on active nodes if using local embeddings.
    # Let's assume the GNN will use nn.Embedding(num_global_vital_concepts, embed_dim),
    # so we just need to collect edge connections using global_concept_ids.

    # --- 3. Create Edges (E_temp, E_meas, E_diag/med/proc) and populate T_t vital features ---
    # E_temp: ('timeslice', 'temporally_precedes', 'timeslice')
    if num_time_bins > 1:
        src_tt_temp = torch.arange(0, num_time_bins - 1, dtype=torch.long)
        dst_tt_temp = torch.arange(1, num_time_bins, dtype=torch.long)
        data['timeslice', 'temporally_precedes', 'timeslice'].edge_index = torch.stack([src_tt_temp, dst_tt_temp], dim=0)
        delta_t_attr = torch.full((src_tt_temp.size(0), 1), float(time_bin_size_hours)) # Edge feature: Δt
        data['timeslice', 'temporally_precedes', 'timeslice'].edge_attr = delta_t_attr

    # Initialize lists for edge data
    edge_lists = {etype: {'src': [], 'dst': [], 'attr': []} for etype in EDGE_TYPES if etype[1] != 'temporally_precedes'}

    # Temporary storage for vital values per T_t bin for aggregation
    # Structure: t_bin_vital_values[time_bin_idx][vital_feature_idx] = [list of values]
    t_bin_vital_values = [[[] for _ in range(num_vital_features)] for _ in range(num_time_bins)]

    # Iterate through patient events (rows in patient_df)
    for _, row in patient_df.iterrows():
        event_time_rel = row['time_rel_hours']
        if pd.isna(event_time_rel):
            continue

        time_bin_idx = int(np.floor(event_time_rel / time_bin_size_hours))
        if not (0 <= time_bin_idx < num_time_bins):
            continue # Event outside the valid binned time range for the snapshot

        # Process Vitals/Labs (E_meas)
        for vital_idx, vital_name in enumerate(vital_col_names):
            if vital_name in row and pd.notna(row[vital_name]):
                measurement_value = float(row[vital_name])
                # TODO: Z-scoring: requires global means/stds for each vital. For now, use raw.
                # z_value = (measurement_value - global_means[vital_name]) / global_stds[vital_name]

                if vital_name not in global_concept_mappers['vital_to_id']:
                    continue # Should not happen if vital_col_names are keys of vital_to_id
                vital_global_id = global_concept_mappers['vital_to_id'][vital_name]

                # Edge: timeslice -> has_vital_measurement -> vital
                edge_lists[('timeslice', 'has_vital_measurement', 'vital')]['src'].append(time_bin_idx)
                edge_lists[('timeslice', 'has_vital_measurement', 'vital')]['dst'].append(vital_global_id)
                edge_lists[('timeslice', 'has_vital_measurement', 'vital')]['attr'].append([measurement_value]) # Edge feature

                # Reverse Edge: vital -> vital_measured_in -> timeslice
                edge_lists[('vital', 'vital_measured_in', 'timeslice')]['src'].append(vital_global_id)
                edge_lists[('vital', 'vital_measured_in', 'timeslice')]['dst'].append(time_bin_idx)
                # No attribute for reverse, or could be same if needed by GNN conv.

                if 0 <= vital_idx < num_vital_features: # Safety check
                     t_bin_vital_values[time_bin_idx][vital_idx].append(measurement_value)

        # Helper function to process discrete codes (D, M, P)
        def process_codes(code_col_name, code_type_str, mapper_key, patient_row, current_time_bin_idx):
            if code_col_name in patient_row and pd.notna(patient_row[code_col_name]):
                codes_raw = patient_row[code_col_name]
                codes_list = []
                if isinstance(codes_raw, str): # Assuming ;-separated if string
                    codes_list = [code.strip() for code in codes_raw.split(';') if code.strip()]
                elif isinstance(codes_raw, (list, np.ndarray)): # If already a list/array of codes
                    codes_list = [str(code).strip() for code in codes_raw if pd.notna(code) and str(code).strip()]
                elif pd.notna(codes_raw): # Single code entry
                    codes_list = [str(codes_raw).strip()]

                for code_str in codes_list:
                    if code_str in global_concept_mappers[mapper_key]:
                        code_global_id = global_concept_mappers[mapper_key][code_str]

                        # Edge: timeslice -> has_<type> -> <type>
                        fwd_edge = ('timeslice', f'has_{code_type_str}', code_type_str)
                        edge_lists[fwd_edge]['src'].append(current_time_bin_idx)
                        edge_lists[fwd_edge]['dst'].append(code_global_id)
                        # edge_lists[fwd_edge]['attr'].append(...) # If discrete events have values

                        # Reverse Edge: <type> -> <type>_active_in -> timeslice
                        rev_edge = (code_type_str, f'{code_type_str}_active_in', 'timeslice')
                        edge_lists[rev_edge]['src'].append(code_global_id)
                        edge_lists[rev_edge]['dst'].append(current_time_bin_idx)

        process_codes(diagnosis_col_name, 'diagnosis', 'diagnosis_to_id', row, time_bin_idx)
        # Commenting out med/proc processing as they are optional and schema might exclude them
        # if medication_col_name and global_concept_mappers.get('medication_to_id'):
        #     process_codes(medication_col_name, 'medication', 'medication_to_id', row, time_bin_idx)
        # if procedure_col_name and global_concept_mappers.get('procedure_to_id'):
        #     process_codes(procedure_col_name, 'procedure', 'procedure_to_id', row, time_bin_idx)

    # Aggregate vital features for T_t nodes (e.g., mean) and update data['timeslice'].x
    temp_ts_features = data['timeslice'].x[:, time_embedding_dim:].clone() # Get the part for vital features
    for bin_idx in range(num_time_bins):
        for vital_feature_idx in range(num_vital_features):
            values_in_bin_for_vital = t_bin_vital_values[bin_idx][vital_feature_idx]
            if values_in_bin_for_vital:
                mean_val = np.mean(values_in_bin_for_vital)
                temp_ts_features[bin_idx, vital_feature_idx] = float(mean_val)
            # NaNs will remain if no values, or if initialized with NaN

    # Replace NaNs with 0 (or another imputation strategy)
    temp_ts_features = torch.nan_to_num(temp_ts_features, nan=0.0, posinf=0.0, neginf=0.0)
    data['timeslice'].x = torch.cat([data['timeslice'].x[:, :time_embedding_dim], temp_ts_features], dim=1)


    # Assign collected edges to HeteroData
    for edge_type_tuple, L in edge_lists.items():
        if L['src']: # If any edges of this type were added
            data[edge_type_tuple].edge_index = torch.tensor([L['src'], L['dst']], dtype=torch.long)
            if L['attr']: # If attributes were collected for this edge type
                # Ensure all attributes have consistent dimension, e.g. by padding or careful construction
                # Assuming all attributes for an edge type are lists of numbers (e.g. [[val1], [val2]])
                try:
                    data[edge_type_tuple].edge_attr = torch.tensor(L['attr'], dtype=torch.float)
                except Exception as e_attr:
                    print(f"Error converting edge_attr for {edge_type_tuple} to tensor: {e_attr}. Attrs: {L['attr'][:5]}")
                    # Fallback or error handling for edge attributes
                    data[edge_type_tuple].edge_attr = torch.empty((len(L['src']), 0)) # No attributes if conversion fails

    # Set number of nodes for concept types based on global mapper size
    # This is crucial for nn.Embedding layers in the GNN model if they use these global IDs.
    data['vital'].num_nodes = len(global_concept_mappers['vital_to_id'])
    data['diagnosis'].num_nodes = len(global_concept_mappers['diagnosis_to_id'])
    if 'medication' in NODE_TYPES: # Only set if node type is active in schema
        data['medication'].num_nodes = len(global_concept_mappers.get('medication_to_id', {}))
    if 'procedure' in NODE_TYPES: # Only set if node type is active in schema
        data['procedure'].num_nodes = len(global_concept_mappers.get('procedure_to_id', {}))

        

    # Node features for concept nodes (V,D,M,P) are not set as 'x' here.
    # Instead, the GNN model will use nn.Embedding layers, taking node IDs as input.
    # If we wanted to store the global IDs as a feature, we could do:
    # data['vital'].global_ids = torch.arange(len(global_concept_mappers['vital_to_id']))
    # But typically, PyG models with embeddings work directly off the node indices up to num_nodes.

    # Ensure all node and edge types from schema are initialized in HeteroData, even if empty.
    # This is important for batching and HeteroConv layers.
    for node_type in NODE_TYPES:
        if node_type not in data.node_types: # Should have been added if num_time_bins > 0 or mappers exist
            data[node_type].num_nodes = 0
            # PyG might require .x to exist, even if empty with 0 features or a defined feature count
            # For now, let's assume if num_nodes is 0, .x might not be strictly needed by all layers,
            # or the GNN model's embedding layers will handle it.
            # If feature_dim for a node type is fixed (e.g. embedding dim), initialize .x with (0, feature_dim)
            # Example: data[node_type].x = torch.empty((0, default_embedding_dim))

    for edge_type_tuple in EDGE_TYPES:
        if edge_type_tuple not in data.edge_types:
            # Initialize with empty edge_index (shape [2,0])
            data[edge_type_tuple].edge_index = torch.empty((2,0), dtype=torch.long)
            # If this edge type is expected to have edge_attr, initialize that too.
            # data[edge_type_tuple].edge_attr = torch.empty((0, num_edge_features_for_this_type))

    # 4. Assign Label - This will be done in the Dataset's get() method after this function returns,
    # as the Dataset has access to the y_series. build_patient_graph focuses on graph structure.

    # Validate the created graph (optional, good for debugging during development)
    try:
        data.validate(raise_on_error=False) # Set to True to halt on error
    except Exception as e_val:
        print(f"HeteroData validation warning for patient {patient_id}: {e_val}")
        # Depending on severity, might return None or the potentially problematic graph
        # For now, we'll return the graph and let downstream processes handle it.

    return data


if __name__ == '__main__':
    # This function cannot be easily tested standalone without mock data matching assumptions
    # and fully implemented global_concept_mappers and patient_df preprocessing.
    print("graph_constructor.py outlined.")
    print("Testing build_patient_graph would require mock data and mappers.")

    # Example of creating a very simple HeteroData object (for syntax reference)
    # hdata = HeteroData()
    # hdata['paper'].x = torch.randn(10, 32) # 10 paper nodes, 32 features
    # hdata['author'].x = torch.randn(5, 16)  # 5 author nodes, 16 features
    # hdata['paper', 'cites', 'paper'].edge_index = torch.randint(0, 10, (2, 20))
    # hdata['author', 'writes', 'paper'].edge_index = torch.stack([
    #     torch.tensor([0,1,2,3,4]), # author indices
    #     torch.tensor([0,2,4,6,8])  # paper indices
    # ], dim=0)
    # print("\nSample HeteroData structure:")
    # print(hdata)
    # hdata.validate(raise_on_error=True)
    pass
