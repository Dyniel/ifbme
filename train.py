import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Attempt to import project-specific modules
try:
    from models.main_model import ModelWithSTMGNNLayer
    from models.stm_gnn import STMGNN
    from models.data_utils import (
        CustomGraphStaticDataset, create_static_graph_dataloaders,
        CustomGraphDynamicDataset, create_dynamic_graph_dataloaders,
        PYG_AVAILABLE_DATAUTILS, Dataset as DatasetPlaceholder, DataLoader as DataLoaderPlaceholder # import placeholders too
    )
    PROJECT_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the root of the repository and all model files exist.")
    PROJECT_MODULES_AVAILABLE = False
    # Define dummy classes if modules are not found, to allow script to be parsable
    class ModelWithSTMGNNLayer(torch.nn.Module):
        def __init__(self, num_features, hidden_dim_stm, num_classes, **kwargs): super().__init__(); self.fc = torch.nn.Linear(hidden_dim_stm,num_classes); self.input_embed = torch.nn.Linear(num_features, hidden_dim_stm)
        def forward(self, x, edge_index, **kwargs): x_emb = self.input_embed(x); return self.fc(x_emb.mean(dim=0, keepdim=True) if x_emb.dim() > 1 else x_emb) # Adjusted dummy output

    class STMGNN(torch.nn.Module):
        def __init__(self, num_node_features, layer_hidden_dim, num_classes, **kwargs): super().__init__(); self.fc = torch.nn.Linear(layer_hidden_dim,num_classes); self.input_embed = torch.nn.Linear(num_node_features,layer_hidden_dim)
        def forward(self, graph_snapshots, **kwargs):
            # graph_snapshots is list of (x, edge_index, time_idx)
            # Use first snapshot's x for dummy processing
            if not graph_snapshots: return torch.randn(1, self.fc.out_features)
            x_first_snap = graph_snapshots[0][0]
            x_emb = self.input_embed(x_first_snap)
            return self.fc(x_emb.mean(dim=0, keepdim=True) if x_emb.dim() > 1 else x_emb) # Adjusted dummy output


    class DatasetPlaceholder: # Simplified for this context
        def __init__(self, is_dynamic=False, num_samples=10, avg_nodes=10, num_features=10, num_classes=2, avg_snapshots=3, **kwargs):
            self.is_dynamic = is_dynamic
            self.num_samples=num_samples
            self.avg_nodes = avg_nodes
            self.num_features = num_features
            self.num_classes = num_classes
            self.avg_snapshots = avg_snapshots
        def __len__(self): return self.num_samples
        def __getitem__(self,idx):
            if self.is_dynamic:
                num_snaps = np.random.randint(1, self.avg_snapshots + 2)
                snaps = []
                for t_idx in range(num_snaps):
                    num_n = np.random.randint(1, self.avg_nodes + 5)
                    snaps.append({'x':torch.randn(num_n, self.num_features),'edge_index':torch.zeros(2,max(0,num_n-1)).long(),'t':torch.tensor(t_idx)})
                return (snaps, torch.randint(0,self.num_classes,(1,)).item())
            else: # Static
                num_n = np.random.randint(1, self.avg_nodes + 5)
                # Mimic PyG Data object structure for placeholder
                return {'x':torch.randn(num_n, self.num_features),'edge_index':torch.zeros(2,max(0,num_n-1)).long(),'y':torch.randint(0,self.num_classes,(1,)).squeeze().item(), 'num_nodes':num_n}


    class DataLoaderPlaceholder: # Simplified
        def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))
        def __iter__(self):
            self.current_pos = 0
            if self.shuffle: np.random.shuffle(self.indices)
            return self
        def __len__(self):
            return (len(self.dataset) + self.batch_size -1) // self.batch_size
        def __next__(self):
            if self.current_pos >= len(self.indices): raise StopIteration

            batch_indices = self.indices[self.current_pos : self.current_pos + self.batch_size]
            self.current_pos += self.batch_size
            items = [self.dataset[i] for i in batch_indices]

            if not items: raise StopIteration

            if not self.dataset.is_dynamic:
                class BatchMimic: # Mimics PyG Batch for static graphs
                    def __init__(self, items_list, device):
                        self.x = torch.cat([i['x'] for i in items_list], dim=0).to(device)
                        # Simplistic edge_index: take first, or make empty if problem
                        self.edge_index = items_list[0]['edge_index'].to(device) if items_list[0]['edge_index'].numel() > 0 else torch.empty(2,0,dtype=torch.long).to(device)
                        self.y = torch.tensor([i['y'] for i in items_list]).to(device)
                        self.num_graphs = len(items_list)
                        node_counts = [i['x'].shape[0] for i in items_list]
                        self.batch = torch.cat([torch.full((nc,), j, dtype=torch.long) for j, nc in enumerate(node_counts)]).to(device)
                    def to(self, device_ignored): # Already on device
                        return self
                return BatchMimic(items, DEVICE) # Pass DEVICE to BatchMimic
            return items # List of (snapshots, label) for dynamic

    PYG_AVAILABLE_DATAUTILS = False

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "model_with_stmgnnlayer" # "stm_gnn" or "model_with_stmgnnlayer"
# MODEL_TYPE = "stm_gnn" # Uncomment to test STMGNN
LEARNING_RATE = 0.001
BATCH_SIZE = 8
EPOCHS = 3

# Static graph dataset parameters
NUM_SAMPLES_STATIC = 50
AVG_NODES_STATIC = 10
FEATURES_STATIC = 16
CLASSES_STATIC = 3

# Dynamic graph dataset parameters
NUM_SEQUENCES_DYNAMIC = 20
AVG_SNAPSHOTS_DYNAMIC = 3
AVG_NODES_DYNAMIC = 8
FEATURES_DYNAMIC = 16
CLASSES_DYNAMIC = 2


def train_epoch_static(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0; correct_predictions = 0; total_samples = 0
    for batch in loader:
        # batch is already on DEVICE if using BatchMimic, or PyG Batch
        # For PyG Batch, batch.to(DEVICE) is needed if not done by loader.
        # Our BatchMimic puts data on DEVICE in its __init__.
        # PyG's default loader does not move to device.
        if PYG_AVAILABLE_DATAUTILS: batch = batch.to(DEVICE)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct_predictions += pred.eq(batch.y).sum().item()
        total_samples += batch.num_graphs
    return (total_loss / total_samples if total_samples > 0 else 0), \
           (correct_predictions / total_samples if total_samples > 0 else 0)

def evaluate_epoch_static(model, loader, criterion):
    model.eval()
    total_loss = 0; correct_predictions = 0; total_samples = 0
    with torch.no_grad():
        for batch in loader:
            if PYG_AVAILABLE_DATAUTILS: batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct_predictions += pred.eq(batch.y).sum().item()
            total_samples += batch.num_graphs
    return (total_loss / total_samples if total_samples > 0 else 0), \
           (correct_predictions / total_samples if total_samples > 0 else 0)


def train_epoch_dynamic(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0; correct_predictions = 0; total_sequences = 0
    for batch_of_sequences in loader: # This is a list of (sequence_data, label)
        if not batch_of_sequences: continue

        optimizer.zero_grad()
        batch_accumulated_loss = torch.tensor(0.0, device=DEVICE)
        num_sequences_in_batch = 0

        for sequence_data_item in batch_of_sequences:
            snapshots_list_of_dicts = sequence_data_item[0] # list of dicts or Data
            label_scalar = sequence_data_item[1]
            label_tensor = torch.tensor([label_scalar], device=DEVICE)

            processed_snapshots_for_model = []
            for snap_dict_or_data in snapshots_list_of_dicts:
                x = snap_dict_or_data['x'].to(DEVICE) if isinstance(snap_dict_or_data, dict) else snap_dict_or_data.x.to(DEVICE)
                ei = snap_dict_or_data['edge_index'].to(DEVICE) if isinstance(snap_dict_or_data, dict) else snap_dict_or_data.edge_index.to(DEVICE)
                tval = snap_dict_or_data['t'].item() if isinstance(snap_dict_or_data, dict) else snap_dict_or_data.t.item()
                processed_snapshots_for_model.append((x, ei, tval))

            if not processed_snapshots_for_model: continue

            out = model(processed_snapshots_for_model) # Expected: [num_classes]
            loss = criterion(out.unsqueeze(0), label_tensor) # out: [C], label: [1]
            batch_accumulated_loss += loss # Accumulate loss for the batch

            if out.argmax(dim=0) == label_tensor.item(): correct_predictions += 1
            num_sequences_in_batch +=1

        if num_sequences_in_batch > 0:
            avg_batch_loss = batch_accumulated_loss / num_sequences_in_batch
            avg_batch_loss.backward()
            optimizer.step()
            total_loss += avg_batch_loss.item() * num_sequences_in_batch # Store sum of losses
            total_sequences += num_sequences_in_batch

    return (total_loss / total_sequences if total_sequences > 0 else 0), \
           (correct_predictions / total_sequences if total_sequences > 0 else 0)


def evaluate_epoch_dynamic(model, loader, criterion):
    model.eval()
    total_loss = 0; correct_predictions = 0; total_sequences = 0
    with torch.no_grad():
        for batch_of_sequences in loader:
            if not batch_of_sequences: continue
            for sequence_data_item in batch_of_sequences:
                snapshots_list_of_dicts = sequence_data_item[0]
                label_scalar = sequence_data_item[1]
                label_tensor = torch.tensor([label_scalar], device=DEVICE)

                processed_snapshots_for_model = []
                for snap_dict_or_data in snapshots_list_of_dicts:
                    x = snap_dict_or_data['x'].to(DEVICE) if isinstance(snap_dict_or_data, dict) else snap_dict_or_data.x.to(DEVICE)
                    ei = snap_dict_or_data['edge_index'].to(DEVICE) if isinstance(snap_dict_or_data, dict) else snap_dict_or_data.edge_index.to(DEVICE)
                    tval = snap_dict_or_data['t'].item() if isinstance(snap_dict_or_data, dict) else snap_dict_or_data.t.item()
                    processed_snapshots_for_model.append((x, ei, tval))

                if not processed_snapshots_for_model: continue
                out = model(processed_snapshots_for_model)
                loss = criterion(out.unsqueeze(0), label_tensor)
                total_loss += loss.item()
                if out.argmax(dim=0) == label_tensor.item(): correct_predictions += 1
                total_sequences += 1
    return (total_loss / total_sequences if total_sequences > 0 else 0), \
           (correct_predictions / total_sequences if total_sequences > 0 else 0)


def main():
    global PYG_AVAILABLE_DATAUTILS # Allow main to modify this based on actual imports
    try:
        from models.data_utils import PYG_AVAILABLE_DATAUTILS as PYG_STATUS_FROM_UTILS
        PYG_AVAILABLE_DATAUTILS = PYG_STATUS_FROM_UTILS
    except ImportError: # If models.data_utils itself is missing
        PYG_AVAILABLE_DATAUTILS = False


    if not PROJECT_MODULES_AVAILABLE: # If models themselves are missing
        print("Exiting: Essential project model modules could not be imported.")
        # Setup dummy versions if proceeding for a dry run of train.py structure
        global ModelWithSTMGNNLayer, STMGNN, CustomGraphStaticDataset, create_static_graph_dataloaders
        global CustomGraphDynamicDataset, create_dynamic_graph_dataloaders

        # Ensure dummy classes defined at top are used
        print("Continuing with top-level placeholder model and data classes for train.py basic execution.")

    # Override data utilities with placeholders if PyG is not available
    if not PYG_AVAILABLE_DATAUTILS:
        print("PyTorch Geometric not available in data_utils. Using placeholder data utilities for train.py.")
        global CustomGraphStaticDataset, create_static_graph_dataloaders
        global CustomGraphDynamicDataset, create_dynamic_graph_dataloaders
        CustomGraphStaticDataset = lambda **kwargs: DatasetPlaceholder(**kwargs, is_dynamic=False)
        CustomGraphDynamicDataset = lambda **kwargs: DatasetPlaceholder(**kwargs, is_dynamic=True)
        # DataLoaderPlaceholder already assigned if initial import failed
        create_static_graph_dataloaders = lambda dataset, batch_size, **kwargs: (DataLoaderPlaceholder(dataset, batch_size), DataLoaderPlaceholder(dataset, batch_size), DataLoaderPlaceholder(dataset, batch_size))
        create_dynamic_graph_dataloaders = lambda dataset, batch_size, **kwargs: (DataLoaderPlaceholder(dataset, batch_size), DataLoaderPlaceholder(dataset, batch_size), DataLoaderPlaceholder(dataset, batch_size))


    print(f"Using device: {DEVICE}")
    print(f"Selected model type: {MODEL_TYPE}")
    print(f"PYG_AVAILABLE_DATAUTILS (after check): {PYG_AVAILABLE_DATAUTILS}")

    model = None; train_loader, val_loader, test_loader = None, None, None

    if MODEL_TYPE == "model_with_stmgnnlayer":
        if not PYG_AVAILABLE_DATAUTILS and PROJECT_MODULES_AVAILABLE:
            print("Warning: 'model_with_stmgnnlayer' training needs PyTorch Geometric for data. Placeholders may not reflect true performance.")
            # Allow to proceed with placeholders if PROJECT_MODULES_AVAILABLE is true but PyG is false.
            # If models are also placeholders, this is fine for a dry run.

        dataset = CustomGraphStaticDataset(
            num_samples=NUM_SAMPLES_STATIC, avg_nodes=AVG_NODES_STATIC, num_features=FEATURES_STATIC,
            num_classes=CLASSES_STATIC, random_seed=42
        )
        train_loader, val_loader, test_loader = create_static_graph_dataloaders(
            dataset, BATCH_SIZE, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        model = ModelWithSTMGNNLayer( # Dummy or real based on PROJECT_MODULES_AVAILABLE
            num_features=FEATURES_STATIC, hidden_dim_stm=64, num_classes=CLASSES_STATIC,
            num_stm_layers=2, heads_stm=2, dropout_stm=0.1,
            global_memory_dim_stm=32, num_memory_slots_stm=3
        ).to(DEVICE)
    elif MODEL_TYPE == "stm_gnn":
        dataset = CustomGraphDynamicDataset(
            num_sequences=NUM_SEQUENCES_DYNAMIC, avg_snapshots=AVG_SNAPSHOTS_DYNAMIC,
            avg_nodes_per_snapshot=AVG_NODES_DYNAMIC, num_features=FEATURES_DYNAMIC,
            num_classes=CLASSES_DYNAMIC, random_seed=123
        )
        train_loader, val_loader, test_loader = create_dynamic_graph_dataloaders(
            dataset, BATCH_SIZE, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123
        )
        model = STMGNN( # Dummy or real
            num_node_features=FEATURES_DYNAMIC, layer_hidden_dim=64, gnn_output_dim=64,
            num_gnn_layers=2, global_memory_dim=32, num_memory_slots=3,
            num_heads=2, dropout=0.1, num_classes=CLASSES_DYNAMIC
        ).to(DEVICE)
    else:
        print(f"Unknown model type: {MODEL_TYPE}"); return

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = F.cross_entropy

    print(f"\nStarting training for {MODEL_TYPE}...")
    train_fn = train_epoch_static if MODEL_TYPE == "model_with_stmgnnlayer" else train_epoch_dynamic
    eval_fn = evaluate_epoch_static if MODEL_TYPE == "model_with_stmgnnlayer" else evaluate_epoch_dynamic

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_fn(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Check if test_loader.dataset has items or if it's a placeholder with 0 length effectively
    has_test_data = False
    try:
        if len(test_loader.dataset) > 0: has_test_data = True
    except: # Placeholder might not have dataset attribute in the same way
        if len(test_loader) > 0 : has_test_data = True # Check if loader itself has batches

    if has_test_data:
        test_loss, test_acc = eval_fn(model, test_loader, criterion)
        print(f"Test Results: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    else:
        print("Test loader is empty or invalid, skipping test evaluation.")

if __name__ == '__main__':
    main()
