# data_utils/graph_loader.py

"""
Contains PyTorch Geometric Dataset and DataLoader implementations for patient graphs.
"""
import torch
from torch_geometric.data import Dataset, DataLoader # DataLoader from PyG for HeteroData
import pandas as pd
import os # For potential saving/loading of processed data

from data_utils.graph_constructor import build_patient_graph # The function we are building
from data_utils.graph_schema import NODE_TYPES, EDGE_TYPES # Import schema

# Helper function to create mappers (could be part of a preprocessing script too)
def create_global_mappers(all_patient_data_df, patient_id_col,
                          vital_col_names, diagnosis_col_name,
                          medication_col_name, procedure_col_name,
                          timestamp_col): # timestamp_col needed for some processing
    """
    Creates global mappers for concept nodes from the entire dataset.
    This should be run once.
    """
    mappers = {}

    # Vital/Lab concepts are typically the column names themselves if they are well-defined
    mappers['vital_to_id'] = {name: i for i, name in enumerate(sorted(list(set(vital_col_names))))}

    # For codes (diagnosis, medication, procedure), it depends on how they are stored.
    # Assuming they might be in lists within cells, or delimited strings.
    # This requires careful parsing of the actual data format.

    def extract_unique_codes(df, col_name):
        unique_codes = set()
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for mapper creation.")
            return {}

        # Example: if codes are in lists or ;-separated strings in the column
        for item_list_or_string in df[col_name].dropna():
            if isinstance(item_list_or_string, list):
                for code in item_list_or_string: unique_codes.add(str(code))
            elif isinstance(item_list_or_string, str):
                for code in item_list_or_string.split(';'): # Example delimiter
                    if code.strip(): unique_codes.add(code.strip())
            else: # single code
                 unique_codes.add(str(item_list_or_string))
        return {code: i for i, code in enumerate(sorted(list(unique_codes)))}

    mappers['diagnosis_to_id'] = extract_unique_codes(all_patient_data_df, diagnosis_col_name)
    # Commenting out med/proc mappers as these columns are optional/placeholders
    # mappers['medication_to_id'] = extract_unique_codes(all_patient_data_df, medication_col_name)
    # mappers['procedure_to_id'] = extract_unique_codes(all_patient_data_df, procedure_col_name)

    # Initialize to empty dicts if not created, to prevent KeyErrors later
    if 'medication_to_id' not in mappers:
        mappers['medication_to_id'] = {}
    if 'procedure_to_id' not in mappers:
        mappers['procedure_to_id'] = {}

    print(f"Created mappers: {len(mappers['vital_to_id'])} vitals, "
          f"{len(mappers['diagnosis_to_id'])} diagnoses, "
          f"{len(mappers.get('medication_to_id', {}))} medications, " # Use .get for safety
          f"{len(mappers.get('procedure_to_id', {}))} procedures.") # Use .get for safety
    return mappers


class PatientHeteroGraphDataset(Dataset):
    def __init__(self, root_dir, # For saving/loading processed data
                 patient_df_split, # DataFrame for the current split (e.g., train or test fold)
                 patient_id_col,
                 y_series_split, # Labels for the current split, indexed by patient_id
                 global_concept_mappers, # Pre-computed mappers
                 target_variable_name,
                 label_timestamp_col, # Column in patient_df_split that indicates the time of label event for snapshotting
                 timestamp_col, # Main event timestamp column in patient_df_split
                 time_rel_col_name, # Name for the relative time column (hours since admission)
                 admission_timestamp_col, # Admission timestamp column in patient_df_split
                 graph_construction_params, # Dict of params for build_patient_graph & others
                 vital_col_names, diagnosis_col_name, medication_col_name, procedure_col_name,
                 transform=None, pre_transform=None, pre_filter=None,
                 force_reprocess=False):

        self.patient_df_split = patient_df_split.copy() # Work on a copy
        self.patient_id_col = patient_id_col
        self.y_series_split = y_series_split # Should be Series/DataFrame indexed by patient_id
        self.global_concept_mappers = global_concept_mappers

        self.target_variable_name = target_variable_name
        self.label_timestamp_col = label_timestamp_col # Used to find time for snapshot
        self.timestamp_col = timestamp_col             # Event timestamps
        self.time_rel_col_name = time_rel_col_name
        self.admission_timestamp_col = admission_timestamp_col

        self.graph_construction_params = graph_construction_params
        self.vital_col_names = vital_col_names
        self.diagnosis_col_name = diagnosis_col_name
        self.medication_col_name = medication_col_name
        self.procedure_col_name = procedure_col_name

        self.processed_dir_name = "processed_patient_graphs" # More specific name
        self.force_reprocess = force_reprocess

        # Get unique patient IDs for this specific data split
        if self.patient_id_col not in self.patient_df_split.columns:
            raise ValueError(f"Patient ID column '{self.patient_id_col}' not found in provided DataFrame.")
        self.patient_ids = sorted(self.patient_df_split[self.patient_id_col].unique())

        # Pre-calculate relative time for the current split
        # Convert timestamp columns to datetime
        self.patient_df_split[self.timestamp_col] = pd.to_datetime(self.patient_df_split[self.timestamp_col], errors='coerce')
        self.patient_df_split[self.admission_timestamp_col] = pd.to_datetime(self.patient_df_split[self.admission_timestamp_col], errors='coerce')

        # Drop rows where essential timestamps are NaT after conversion
        self.patient_df_split.dropna(subset=[self.timestamp_col, self.admission_timestamp_col], inplace=True)

        # Calculate relative time ('time_rel_col_name')
        # Group by patient to ensure correct reference to *their own* admission time
        self.patient_df_split[self.time_rel_col_name] = self.patient_df_split.groupby(self.patient_id_col, group_keys=False)\
            .apply(lambda x: (x[self.timestamp_col] - x[self.admission_timestamp_col].min()).dt.total_seconds() / 3600)\
            .reset_index(level=0, drop=True) # Aligns series after groupby

        # Filter patient_ids list based on those still present after NaT drop & available in y_series_split
        valid_pids_after_time_calc = self.patient_df_split[self.patient_id_col].unique()
        self.patient_ids = sorted(list(set(self.patient_ids).intersection(set(valid_pids_after_time_calc))))
        self.patient_ids = [pid for pid in self.patient_ids if pid in self.y_series_split.index]

        if not self.patient_ids:
            print("Warning: No valid patient IDs remaining after initial processing in Dataset __init__.")

        super().__init__(root_dir, transform, pre_transform, pre_filter)

        # Check if processing is needed (PyG's way, adapted)
        # The processed_file_names property depends on self.patient_ids, so call super().__init__ first.
        # Then, decide if self.process() needs to be called.
        # PyG's Dataset constructor calls self.process() if self._check_processed() fails.
        # We can rely on that or be more explicit.
        if self.force_reprocess:
            print(f"Force reprocessing graphs dataset...")
            self._process() # Call PyG's internal method that calls our self.process()
        elif not self._is_processed():
            print(f"Processed graph dataset not found or incomplete. Processing...")
            self._process()
        else:
            print("Found complete processed graph dataset.")

    @property
    def _processed_dir(self): # Helper for consistency
        return os.path.join(self.root, self.processed_dir_name)

    def _is_processed(self):
        # Check if all expected processed files exist for the current patient_ids
        if not os.path.exists(self._processed_dir):
            return False
        if not self.patient_ids: # If no patients, it's "processed"
            return True
        for pid in self.patient_ids:
            if not os.path.exists(os.path.join(self._processed_dir, f'patient_graph_{pid}.pt')):
                return False
        return True

    @property
    def raw_file_names(self):
        # This dataset assumes raw data (patient_df, y_series) is passed directly.
        # If we were loading from specific files, they'd be listed here.
        return [] # No specific raw files to check for, data is in-memory

    @property
    def processed_file_names(self):
        # This list tells PyG what to check for. If these files exist, process() is skipped.
        # If force_reprocess is True, we want to bypass this check effectively.
        # PyG's internal logic for this is a bit tricky with force_reprocess.
        # The self._is_processed() check in __init__ handles it more directly.
        # Returning a generic name if empty, or specific names.
        if not self.patient_ids: return ["dummy_empty_dataset_marker.pt"] # Handles empty dataset case
        return [os.path.join(self.processed_dir_name, f'patient_graph_{pid}.pt') for pid in self.patient_ids]


    def download(self):
        pass # Data passed in memory

    def process(self):
        if not os.path.exists(self._processed_dir):
            os.makedirs(self._processed_dir)

        processed_count = 0
        for patient_id in self.patient_ids:
            # patient_specific_df already has 'time_rel_col_name' calculated in __init__
            # We need to get the correct slice of the main df for this patient.
            # self.patient_df_split contains all data for the current split, with time_rel_col_name
            current_patient_df_full_history = self.patient_df_split[self.patient_df_split[self.patient_id_col] == patient_id]

            if current_patient_df_full_history.empty:
                print(f"Warning: No data found for patient_id {patient_id} in patient_df_split. Skipping.")
                continue

            if patient_id not in self.y_series_split.index: # Check against the y_series_split index
                print(f"Warning: Label info not found for patient {patient_id} in y_series_split. Skipping.")
                continue

            label = self.y_series_split.loc[patient_id] # Assuming y_series_split is indexed by patient_id

            # Determine label_timestamp_abs for snapshotting
            # This requires the label_timestamp_col to exist in current_patient_df_full_history
            # and to correspond to the event defining the label (e.g., discharge/death event time)
            # Using the *last available* timestamp in label_timestamp_col for the patient as a proxy.
            # This needs to be very robust in a real application.
            if self.label_timestamp_col not in current_patient_df_full_history.columns:
                 print(f"Warning: Label timestamp column '{self.label_timestamp_col}' not found for patient {patient_id}. Using last event time as proxy.")
                 # Fallback: use the timestamp of the patient's last recorded event in the current split
                 # This assumes timestamp_col is already datetime
                 label_ts_abs = current_patient_df_full_history[self.timestamp_col].max()
            else:
                 label_ts_from_col = current_patient_df_full_history[self.label_timestamp_col].dropna()
                 if not label_ts_from_col.empty:
                     label_ts_abs = pd.to_datetime(label_ts_from_col.iloc[0]) # Use first non-null, assumes it's consistent or unique
                 else:
                     print(f"Warning: Label timestamp in '{self.label_timestamp_col}' is null for patient {patient_id}. Using last event time.")
                     label_ts_abs = current_patient_df_full_history[self.timestamp_col].max()

            if pd.isna(label_ts_abs): # Ensure label_timestamp is valid
                 print(f"Warning: Determined label_timestamp is NaT for patient {patient_id}. Skipping graph construction.")
                 continue

            # --- Snapshotting logic based on absolute label_ts_abs and relative time ---
            # Convert absolute label_ts_abs to relative time for filtering
            patient_admission_time_abs = current_patient_df_full_history[self.admission_timestamp_col].min()
            if pd.isna(patient_admission_time_abs):
                print(f"Warning: Admission time is NaT for patient {patient_id}. Cannot determine snapshot. Skipping.")
                continue

            snapshot_end_time_rel = (label_ts_abs - patient_admission_time_abs).total_seconds() / 3600

            # Filter events that occurred strictly before the snapshot_end_time_rel
            patient_df_snap = current_patient_df_full_history[current_patient_df_full_history[self.time_rel_col_name] < snapshot_end_time_rel].copy()

            if self.graph_construction_params.get('max_graph_history_hours') is not None:
                history_limit_rel = snapshot_end_time_rel - self.graph_construction_params['max_graph_history_hours']
                # Ensure we don't go before admission (relative time 0)
                history_limit_rel = max(0, history_limit_rel)
                patient_df_snap = patient_df_snap[patient_df_snap[self.time_rel_col_name] >= history_limit_rel]

            if patient_df_snap.empty:
                print(f"Warning: Patient {patient_id} has no data in the final snapshot window before {snapshot_end_time_rel:.2f} rel_hours. Skipping graph.")
                continue

            graph_data = build_patient_graph(
                patient_df=patient_df_snap, # Pass the snapshotted DataFrame
                patient_id=patient_id,
                target_variable_name=self.target_variable_name,
                label_timestamp=label_ts_abs, # For reference, primarily used for snapshotting above
                time_bin_size_hours=self.graph_construction_params.get('time_bin_size_hours', 1),
                vital_col_names=self.vital_col_names,
                diagnosis_col_name=self.diagnosis_col_name,
                medication_col_name=self.medication_col_name,
                procedure_col_name=self.procedure_col_name,
                global_concept_mappers=self.global_concept_mappers,
                max_graph_history_hours=self.graph_construction_params.get('max_graph_history_hours') # build_patient_graph might use this for internal checks
            )

            if graph_data is not None:
                graph_data.y = torch.tensor([label], dtype=torch.float) # Or long if classification with CrossEntropy

                if self.pre_filter is not None and not self.pre_filter(graph_data):
                    continue
                if self.pre_transform is not None:
                    graph_data = self.pre_transform(graph_data)

                save_path = os.path.join(self._processed_dir, f'patient_graph_{patient_id}.pt')
                torch.save(graph_data, save_path)
                processed_count += 1
            else:
                print(f"Graph construction returned None for patient {patient_id}. Skipping save.")

        if processed_count == 0 and self.patient_ids:
             print(f"Warning: No graphs were successfully processed and saved for {len(self.patient_ids)} patients in this split.")
        elif not self.patient_ids and not os.path.exists(os.path.join(self._processed_dir, "dummy_empty_dataset_marker.pt")):
            # Handle empty dataset case for processed_file_names check by PyG
            torch.save(HeteroData(), os.path.join(self._processed_dir, "dummy_empty_dataset_marker.pt"))


    def len(self):
        return len(self.patient_ids)

    def get(self, idx):
        # Loads a single processed graph.
        if idx >= len(self.patient_ids): # Should not happen if DataLoader is configured correctly
            raise IndexError(f"Index {idx} out of bounds for {len(self.patient_ids)} patients.")

        patient_id = self.patient_ids[idx]
        file_path = os.path.join(self._processed_dir, f'patient_graph_{patient_id}.pt')
        try:
            data = torch.load(file_path)
        except FileNotFoundError:
            print(f"ERROR: Processed file for patient {patient_id} at {file_path} not found. Did process() run correctly and save files?")
            # This indicates a critical issue if process() was supposed to generate this file.
            # Re-running process() inside get() is generally not a good idea for performance.
            raise RuntimeError(f"Processed graph for patient {patient_id} not found. Please ensure dataset is processed and paths are correct.")
        return data

if __name__ == '__main__':
    print("PatientHeteroGraphDataset outlined.")
    print("Testing this class requires mock data setup similar to a real scenario,")
    print("including patient_df, y_series, mappers, and column name definitions.")
    # Example instantiation (conceptual, won't run without actual data & params)
    # params = {
    #     'time_bin_size_hours': 1,
    #     'max_graph_history_hours': 24
    # }
    # Assuming dummy_full_df, patient_id_col_name, y_series_dummy, etc. are defined
    # dataset = PatientHeteroGraphDataset(
    #     root_dir='./temp_processed_graphs',
    #     raw_patient_df=None, # dummy_full_df,
    #     patient_id_col=None, # patient_id_col_name,
    #     y_series=None, # y_series_dummy,
    #     target_variable_name = 'outcome',
    #     label_timestamp_col = 'discharge_time', # Example
    #     timestamp_col = 'event_timestamp', # Example
    #     time_rel_col = 'hours_since_admission',
    #     admission_timestamp_col = 'admission_time', # Example
    #     graph_construction_params=params,
    #     vital_col_names=[], diagnosis_col_name=None, medication_col_name=None, procedure_col_name=None,
    #     reprocess_graphs=True # Force processing for test
    # )
    # if len(dataset) > 0:
    #     print(f"Dataset created with {len(dataset)} graphs.")
    #     sample_graph = dataset[0]
    #     print("Sample graph from dataset:")
    #     print(sample_graph)
    #     # PyG DataLoader
    #     # pyg_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #     # for batch in pyg_dataloader:
    #     #     print("Batch from PyG DataLoader:")
    #     #     print(batch)
    #     #     break
    # else:
    #     print("Dataset created but is empty (or all patients skipped during processing).")

    pass
