import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TabularSequenceDataset(Dataset):
    """
    Dataset class for tabular data to be treated as sequences (e.g., for TECO-Transformer).
    Each row from the DataFrame is treated as a sequence of length 1.
    """

    def __init__(self, data_frame, targets=None, feature_columns=None, target_column_name=None, ensure_float32=True):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame containing the features.
            targets (pd.Series or np.ndarray, optional): Series or array of target labels.
                                                        If None, target_column_name must be in data_frame.
            feature_columns (list of str, optional): List of column names to be used as features.
                                                     If None, all columns except target_column_name are used.
            target_column_name (str, optional): Name of the target column in data_frame if targets is None.
            ensure_float32 (bool): If True, ensures feature data is cast to float32.
        """
        super().__init__()

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError("data_frame must be a pandas DataFrame.")

        self.ensure_float32 = ensure_float32

        if feature_columns:
            self.feature_columns = [col for col in feature_columns if col in data_frame.columns]
            missing_cols = set(feature_columns) - set(self.feature_columns)
            if missing_cols:
                print(f"Warning: Specified feature columns not found in DataFrame and will be ignored: {missing_cols}")
        else:
            if target_column_name and target_column_name in data_frame.columns:
                self.feature_columns = [col for col in data_frame.columns if col != target_column_name]
            else:
                self.feature_columns = data_frame.columns.tolist()

        if not self.feature_columns:
            raise ValueError(
                "No feature columns determined. Please specify feature_columns or ensure data_frame is not empty.")

        self.features_df = data_frame[self.feature_columns]

        if targets is not None:
            if len(targets) != len(data_frame):
                raise ValueError("Length of targets must match length of data_frame.")
            self.targets = pd.Series(targets) if not isinstance(targets, pd.Series) else targets
        elif target_column_name and target_column_name in data_frame.columns:
            self.targets = data_frame[target_column_name]
        elif target_column_name:  # target_column_name provided but not in df, and targets is None
            raise ValueError(
                f"Target column '{target_column_name}' not found in DataFrame and no explicit targets provided.")
        else:  # No targets provided in any way
            # Create dummy targets if no target info is available (e.g., for inference)
            # print("Warning: No targets or target_column_name provided. Using dummy targets (zeros).")
            self.targets = pd.Series(np.zeros(len(data_frame), dtype=int))

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        """
        Returns a single sample: a sequence of features and its target.
        For tabular data, each row is treated as a sequence of length 1.
        Output shape for features: (1, num_features)
        """
        features_row = self.features_df.iloc[idx].values
        target_label = self.targets.iloc[idx]

        # Convert features to a tensor. For tabular data, this becomes a sequence of length 1.
        # Shape: (1, num_features)
        sequence_tensor = torch.tensor(features_row, dtype=torch.float32 if self.ensure_float32 else None).unsqueeze(0)

        # Target should be a LongTensor for CrossEntropyLoss
        target_tensor = torch.tensor(target_label, dtype=torch.long)

        return {
            "sequence": sequence_tensor,  # (1, num_features)
            "target": target_tensor  # (scalar)
        }


def basic_collate_fn(batch):
    """
    Collate function for DataLoader to handle batches of sequences.
    Each item in 'batch' is a dictionary like {"sequence": tensor, "target": tensor}.

    Args:
        batch (list of dict): A list of samples from TabularSequenceDataset.

    Returns:
        dict: A dictionary containing batched sequences, targets, and padding masks.
              'sequence': (batch_size, max_seq_len, num_features)
              'target': (batch_size)
              'padding_mask': (batch_size, max_seq_len) - True for padded, False for real.
    """
    sequences = [item['sequence'] for item in batch]  # List of (1, num_features) tensors
    targets = torch.stack([item['target'] for item in batch])  # (batch_size)

    # Since all sequences from TabularSequenceDataset are of length 1,
    # padding is straightforward, but we still create the mask for consistency.
    # max_seq_len will be 1 for this specific dataset.
    max_seq_len = max(s.size(0) for s in sequences) if sequences else 0
    num_features = sequences[0].size(1) if sequences and sequences[0].nelement() > 0 else 0

    # Pad sequences to max_seq_len (which is 1 here, so no actual padding needed if all are length 1)
    # And create padding masks
    padded_sequences = torch.zeros(len(batch), max_seq_len, num_features, dtype=torch.float32)
    # Padding mask: True for padded positions, False for actual data
    padding_masks = torch.ones(len(batch), max_seq_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len, :] = seq
        padding_masks[i, :seq_len] = False  # Mark actual data positions as False (not padded)

    return {
        'sequence': padded_sequences,  # (batch_size, 1, num_features)
        'target': targets,  # (batch_size)
        'padding_mask': padding_masks  # (batch_size, 1) - True for padded (none here), False for real
    }


if __name__ == '__main__':
    # Example Usage:
    print("--- TabularSequenceDataset & basic_collate_fn Example ---")

    # 1. Create dummy data
    data = {
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10),
        'outcome': np.random.randint(0, 3, 10)  # 3 classes
    }
    dummy_df = pd.DataFrame(data)
    feature_cols = ['feature1', 'feature2', 'feature3']
    target_col = 'outcome'

    print("Dummy DataFrame head:\n", dummy_df.head(3))

    # 2. Initialize Dataset
    try:
        dataset = TabularSequenceDataset(dummy_df, feature_columns=feature_cols, target_column_name=target_col)
        print(f"\nDataset created. Length: {len(dataset)}")

        # 3. Get a sample
        sample0 = dataset[0]
        print(f"\nSample 0 from dataset:")
        print(f"  Sequence shape: {sample0['sequence'].shape}")  # Expected: (1, num_features)
        print(f"  Sequence dtype: {sample0['sequence'].dtype}")
        print(f"  Target: {sample0['target']}")
        print(f"  Target dtype: {sample0['target'].dtype}")
        assert sample0['sequence'].shape == (1, len(feature_cols))
        assert sample0['target'].ndim == 0  # Scalar tensor

    except Exception as e:
        print(f"Error during dataset creation or sampling: {e}")
        raise

    # 4. Use with DataLoader and collate_fn
    from torch.utils.data import DataLoader

    try:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=basic_collate_fn)
        print("\nDataLoader created.")

        # Get a batch
        batch_data = next(iter(dataloader))
        print("\nSample batch from DataLoader:")
        print(f"  Batched sequences shape: {batch_data['sequence'].shape}")  # Expected: (batch_size, 1, num_features)
        print(f"  Batched targets shape: {batch_data['target'].shape}")  # Expected: (batch_size)
        print(f"  Batched padding mask shape: {batch_data['padding_mask'].shape}")  # Expected: (batch_size, 1)
        print(f"  Example padding mask for first item in batch: {batch_data['padding_mask'][0]}")  # Should be [False]

        assert batch_data['sequence'].shape == (4, 1, len(feature_cols))
        assert batch_data['target'].shape == (4,)
        assert batch_data['padding_mask'].shape == (4, 1)
        assert not batch_data['padding_mask'][0, 0].item(), "Padding mask for sequence of length 1 should be False."

        print("\n--- Example Finished Successfully ---")

    except Exception as e:
        print(f"Error during DataLoader usage: {e}")
        raise
