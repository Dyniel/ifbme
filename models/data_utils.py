import torch
import numpy as np

try:
    from torch_geometric.data import Data, Dataset, DataLoader
    from torch_geometric.utils import erdos_renyi_graph
    PYG_AVAILABLE_DATAUTILS = True
except ImportError:
    PYG_AVAILABLE_DATAUTILS = False
    print("Warning (data_utils.py): PyTorch Geometric not found. Graph Dataset and DataLoader functionality will be placeholders.")

    # Basic Placeholders if PyG is not available
    class DatasetPlaceholder:
        def __init__(self, num_samples=100, num_nodes=10, num_features=16, num_classes=2, is_dynamic=False, num_snapshots=3):
            self.num_samples = num_samples
            self.num_nodes = num_nodes
            self.num_features = num_features
            self.num_classes = num_classes
            self.is_dynamic = is_dynamic
            self.num_snapshots = num_snapshots
            print(f"Initialized DatasetPlaceholder: {num_samples} samples.")

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Return a dummy data structure that mimics what the models might expect
            if self.is_dynamic:
                snapshots = []
                for _ in range(self.num_snapshots):
                    nodes = np.random.randint(self.num_nodes // 2, self.num_nodes * 2)
                    x = torch.randn(nodes, self.num_features)
                    edge_index = torch.empty((2,0), dtype=torch.long) # Placeholder
                    if nodes > 1:
                         # Simplified edge creation
                        num_edges = np.random.randint(nodes, nodes * 2)
                        edge_index = torch.randint(0, nodes, (2, num_edges), dtype=torch.long)
                    snapshots.append({'x': x, 'edge_index': edge_index, 't': _})
                label = torch.randint(0, self.num_classes, (1,)).item()
                return snapshots, label # List of dicts and a label
            else:
                x = torch.randn(self.num_nodes, self.num_features)
                edge_index = torch.empty((2,0), dtype=torch.long)
                if self.num_nodes > 1:
                    num_edges = np.random.randint(self.num_nodes, self.num_nodes * 2)
                    edge_index = torch.randint(0, self.num_nodes, (2, num_edges), dtype=torch.long)

                y = torch.randint(0, self.num_classes, (1,)).squeeze()
                # Mimic PyG Data object with attributes
                class DataPlaceholder:
                    def __init__(self, x, edge_index, y):
                        self.x = x
                        self.edge_index = edge_index
                        self.y = y
                        self.num_nodes = x.shape[0]
                return DataPlaceholder(x=x, edge_index=edge_index, y=y)

    class DataLoaderPlaceholder:
        def __init__(self, dataset, batch_size=32, shuffle=True, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0
            print(f"Initialized DataLoaderPlaceholder: batch_size={batch_size}.")

        def __iter__(self):
            self.current_idx = 0
            if self.shuffle: # Re-shuffle every epoch
                np.random.shuffle(self.indices)
            return self

        def __next__(self):
            if self.current_idx >= len(self.indices):
                raise StopIteration

            batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
            self.current_idx += self.batch_size

            items = [self.dataset[i] for i in batch_indices]

            if not items:
                raise StopIteration

            # This is a very simplified batching for placeholder.
            # PyG DataLoader does sophisticated batching of graph structures.
            # For DataPlaceholder (static graphs):
            if isinstance(items[0], object) and hasattr(items[0], 'x'): # Assuming DataPlaceholder
                # This is not true graph batching. It's just a list of graphs.
                # A real PyG DataLoader would collate these into a single giant graph (Batch object).
                return items
            # For dynamic graph snapshots (list of dicts):
            elif isinstance(items[0], tuple) and isinstance(items[0][0], list):
                 # Returns list of (list_of_snapshots, label)
                return items

            return items # Fallback

    # Make them accessible if PyG not found
    if not PYG_AVAILABLE_DATAUTILS:
        Dataset = DatasetPlaceholder
        DataLoader = DataLoaderPlaceholder


if PYG_AVAILABLE_DATAUTILS:
    class CustomGraphStaticDataset(Dataset):
        """
        A custom dataset for static graphs.
        Each item is a torch_geometric.data.Data object.
        """
        def __init__(self, num_samples=100, avg_nodes=20, avg_degree=0.2, num_features=32, num_classes=2, random_seed=None):
            super(CustomGraphStaticDataset, self).__init__(None, None, None) # root, transform, pre_transform
            self.num_samples = num_samples
            self.avg_nodes = avg_nodes
            self.avg_degree = avg_degree # Edge probability for Erdos-Renyi
            self.num_features = num_features
            self.num_classes = num_classes
            self.random_seed = random_seed
            if self.random_seed:
                torch.manual_seed(self.random_seed)
                np.random.seed(self.random_seed)

            self.data_list = self._generate_data()

        def _generate_data(self):
            data_list = []
            for i in range(self.num_samples):
                num_nodes = np.random.randint(max(2, self.avg_nodes - 10), self.avg_nodes + 10)

                # Node features
                x = torch.randn(num_nodes, self.num_features)

                # Edges (Erdos-Renyi graph)
                edge_index = erdos_renyi_graph(num_nodes, self.avg_degree, directed=False)

                # Label
                y = torch.randint(0, self.num_classes, (1,)).long() # Ensure label is long for CrossEntropyLoss

                data = Data(x=x, edge_index=edge_index, y=y)
                data.sample_id = torch.tensor([i]) # Store sample_id
                data_list.append(data)
            return data_list

        def len(self):
            return len(self.data_list)

        def get(self, idx):
            return self.data_list[idx]

    class CustomGraphDynamicDataset(Dataset):
        """
        A custom dataset for dynamic graphs (sequences of snapshots).
        Each item is a list of torch_geometric.data.Data objects, and a label for the sequence.
        """
        def __init__(self, num_sequences=50, avg_snapshots=5,
                     avg_nodes_per_snapshot=15, avg_degree_per_snapshot=0.25,
                     num_features=32, num_classes=2, random_seed=None):
            super(CustomGraphDynamicDataset, self).__init__(None, None, None)
            self.num_sequences = num_sequences
            self.avg_snapshots = avg_snapshots
            self.avg_nodes_per_snapshot = avg_nodes_per_snapshot
            self.avg_degree_per_snapshot = avg_degree_per_snapshot
            self.num_features = num_features
            self.num_classes = num_classes
            self.random_seed = random_seed
            if self.random_seed:
                torch.manual_seed(self.random_seed)
                np.random.seed(self.random_seed)

            self.sequences = self._generate_sequences()

        def _generate_sequences(self):
            sequences = []
            for i in range(self.num_sequences):
                num_snapshots = np.random.randint(max(1,self.avg_snapshots-2), self.avg_snapshots+2)
                snapshots_list = []
                for t in range(num_snapshots):
                    num_nodes = np.random.randint(max(2, self.avg_nodes_per_snapshot - 5), self.avg_nodes_per_snapshot + 5)
                    x = torch.randn(num_nodes, self.num_features)
                    edge_index = erdos_renyi_graph(num_nodes, self.avg_degree_per_snapshot, directed=False)
                    # PyG Data object for each snapshot (no individual labels y for snapshots here)
                    snapshot_data = Data(x=x, edge_index=edge_index, t=torch.tensor([t]))
                    snapshots_list.append(snapshot_data)

                # Label for the entire sequence
                sequence_label = torch.randint(0, self.num_classes, (1,)).long()
                sequences.append({'snapshots': snapshots_list, 'label': sequence_label, 'sequence_id': torch.tensor([i])})
            return sequences

        def len(self):
            return len(self.sequences)

        def get(self, idx):
            # Returns the list of Data objects (snapshots) and the sequence label
            return self.sequences[idx]


def create_static_graph_dataloaders(full_dataset, batch_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    Creates train, validation, and test DataLoaders for static graph datasets.
    """
    if not PYG_AVAILABLE_DATAUTILS:
        print("Warning: Using DataLoaderPlaceholder as PyTorch Geometric is not available.")
        # Simplified split for placeholder
        train_size = int(train_ratio * len(full_dataset))
        val_size = int(val_ratio * len(full_dataset))
        # test_size = len(full_dataset) - train_size - val_size

        # Placeholder datasets don't support subset easily, so just return DataLoaders on full dataset
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    if not (train_ratio + val_ratio + test_ratio == 1.0): # Check sum before normalization
        print(f"Warning: train, val, test ratios ({train_ratio}, {val_ratio}, {test_ratio}) do not sum to 1. Normalizing.")
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor(val_ratio * dataset_size))

    train_indices = indices[:train_split]
    val_indices = indices[train_split : train_split + val_split]
    test_indices = indices[train_split + val_split :]

    train_dataset = full_dataset[torch.tensor(train_indices)]
    val_dataset = full_dataset[torch.tensor(val_indices)]
    test_dataset = full_dataset[torch.tensor(test_indices)]

    # PyG DataLoader handles batching of Data objects into Batch objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoaders created: Train ({len(train_dataset)} samples), Val ({len(val_dataset)} samples), Test ({len(test_dataset)} samples)")
    return train_loader, val_loader, test_loader


# For dynamic graph datasets, PyG's DataLoader will batch items as they are.
# If each item is a list of snapshots + label, a batch will be a list of these items.
# Collation needs to be handled carefully in the training loop for dynamic graphs.
def create_dynamic_graph_dataloaders(full_dataset, batch_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    Creates train, validation, and test DataLoaders for dynamic graph datasets.
    Note: The default PyG DataLoader will create batches that are lists of (list of snapshots, label).
    The training loop needs to handle this structure.
    """
    if not PYG_AVAILABLE_DATAUTILS:
        print("Warning: Using DataLoaderPlaceholder for dynamic graphs.")
        # Simplified split for placeholder
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    if not (train_ratio + val_ratio + test_ratio == 1.0): # Check sum before normalization
        print(f"Warning: train, val, test ratios ({train_ratio}, {val_ratio}, {test_ratio}) do not sum to 1. Normalizing.")
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor(val_ratio * dataset_size))

    train_indices = indices[:train_split]
    val_indices = indices[train_split : train_split + val_split]
    test_indices = indices[train_split + val_split :]

    # PyTorch Dataset's default collate_fn will be used by DataLoader if dataset returns list/dict
    # This means a batch will be a list of items from dataset.get()
    # e.g., if dataset.get() returns {'snapshots': [...], 'label': ...},
    # a batch will be [{'snapshots': [...], 'label': ...}, {'snapshots': [...], 'label': ...}, ...]
    # This is usually fine for dynamic graph sequences if the model's forward pass can handle a batch of sequences.

    # To use PyG's subset selection, the dataset must support it.
    # Our CustomGraphDynamicDataset returns dicts, so we subset indices manually.

    class SubsetDynamicDataset(Dataset): # Basic subset wrapper
        def __init__(self, main_dataset, indices):
            super(SubsetDynamicDataset, self).__init__(None, None, None)
            self.main_dataset = main_dataset
            self.indices = indices
        def len(self):
            return len(self.indices)
        def get(self, idx):
            return self.main_dataset.get(self.indices[idx])

    train_subset = SubsetDynamicDataset(full_dataset, train_indices)
    val_subset = SubsetDynamicDataset(full_dataset, val_indices)
    test_subset = SubsetDynamicDataset(full_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Dynamic DataLoaders created: Train ({len(train_subset)} sequences), Val ({len(val_subset)} sequences), Test ({len(test_subset)} sequences)")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    if PYG_AVAILABLE_DATAUTILS:
        print("--- Testing Static Graph Dataset and DataLoader ---")
        static_dataset = CustomGraphStaticDataset(num_samples=50, avg_nodes=15, num_features=10, num_classes=2, random_seed=42)
        print(f"Generated static dataset with {len(static_dataset)} graphs.")
        sample_static_graph = static_dataset.get(0)
        print(f"Sample static graph 0: {sample_static_graph}")
        print(f"  x shape: {sample_static_graph.x.shape}")
        print(f"  edge_index shape: {sample_static_graph.edge_index.shape}")
        print(f"  y: {sample_static_graph.y}")
        print(f"  num_nodes: {sample_static_graph.num_nodes}")


        train_s_loader, val_s_loader, test_s_loader = create_static_graph_dataloaders(static_dataset, batch_size=16, seed=42)

        print("\nIterating through one batch of static train_loader:")
        for batch in train_s_loader:
            print(f"Batch type: {type(batch)}")
            print(f"Batch: {batch}") # PyG Batch object
            print(f"  Batch x shape: {batch.x.shape}")
            print(f"  Batch edge_index shape: {batch.edge_index.shape}")
            print(f"  Batch y shape: {batch.y.shape}")
            print(f"  Batch num_graphs: {batch.num_graphs}")
            print(f"  Batch batch vector shape: {batch.batch.shape}") # maps each node to its graph in the batch
            break # Only one batch

        print("\n--- Testing Dynamic Graph Dataset and DataLoader ---")
        dynamic_dataset = CustomGraphDynamicDataset(num_sequences=20, avg_snapshots=4, avg_nodes_per_snapshot=10, num_features=8, num_classes=3, random_seed=123)
        print(f"Generated dynamic dataset with {len(dynamic_dataset)} sequences.")
        sample_dynamic_sequence = dynamic_dataset.get(0)
        print(f"Sample dynamic sequence 0:")
        print(f"  Number of snapshots: {len(sample_dynamic_sequence['snapshots'])}")
        print(f"  First snapshot data: {sample_dynamic_sequence['snapshots'][0]}")
        print(f"  Sequence label: {sample_dynamic_sequence['label']}")

        train_d_loader, val_d_loader, test_d_loader = create_dynamic_graph_dataloaders(dynamic_dataset, batch_size=4, seed=123)
        print("\nIterating through one batch of dynamic train_loader:")
        for dynamic_batch in train_d_loader:
            print(f"Dynamic Batch type: {type(dynamic_batch)}") # Should be a list of dicts (if default collate)
            print(f"Number of sequences in batch: {len(dynamic_batch)}")
            first_item_in_batch = dynamic_batch[0]
            print(f"First item in batch (type {type(first_item_in_batch)}):")
            print(f"  Snapshots type: {type(first_item_in_batch['snapshots'])}")
            print(f"  Number of snapshots in first item: {len(first_item_in_batch['snapshots'])}")
            print(f"  First snapshot of first item: {first_item_in_batch['snapshots'][0]}")
            print(f"  Label of first item: {first_item_in_batch['label']}")
            break # Only one batch
    else:
        print("--- Testing Placeholder Dataset and DataLoader ---")
        static_placeholder_dataset = Dataset(num_samples=50, num_nodes=15, num_features=10, num_classes=2)
        print(f"Generated static placeholder dataset with {len(static_placeholder_dataset)} graphs.")
        sample_placeholder_graph = static_placeholder_dataset[0]
        print(f"Sample static placeholder graph 0: x={sample_placeholder_graph.x.shape}, y={sample_placeholder_graph.y}")

        static_placeholder_loader = DataLoader(static_placeholder_dataset, batch_size=16)
        print("\nIterating through one batch of static placeholder_loader:")
        for batch in static_placeholder_loader:
            print(f"Batch type: {type(batch)}, length: {len(batch)}") # List of DataPlaceholder objects
            print(f"First item in batch: x={batch[0].x.shape}, y={batch[0].y}")
            break

        dynamic_placeholder_dataset = Dataset(num_samples=20, num_nodes=10, num_features=8, num_classes=3, is_dynamic=True, num_snapshots=4)
        print(f"\nGenerated dynamic placeholder dataset with {len(dynamic_placeholder_dataset)} sequences.")
        sample_dynamic_placeholder = dynamic_placeholder_dataset[0] # (list_of_snapshots, label)
        print(f"Sample dynamic placeholder sequence 0: num_snapshots={len(sample_dynamic_placeholder[0])}, label={sample_dynamic_placeholder[1]}")

        dynamic_placeholder_loader = DataLoader(dynamic_placeholder_dataset, batch_size=4)
        print("\nIterating through one batch of dynamic placeholder_loader:")
        for dynamic_batch in dynamic_placeholder_loader: # List of (list_of_snapshots, label)
            print(f"Dynamic Batch type: {type(dynamic_batch)}, num_sequences_in_batch: {len(dynamic_batch)}")
            first_item_in_batch_snaps, first_item_in_batch_label = dynamic_batch[0]
            print(f"First sequence in batch: num_snapshots={len(first_item_in_batch_snaps)}, label={first_item_in_batch_label}")
            break

    print("\n--- data_utils.py example finished ---")

```
