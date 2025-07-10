import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming PyTorch Geometric for GATConv, and our STMGNNLayer
try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning (main_model.py): PyTorch Geometric not found. GATConv will be a placeholder.")
    # Placeholder GATConv if PyG is not available
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, **kwargs):
            super(GATConv, self).__init__()
            effective_out_channels = out_channels * heads if concat else out_channels
            self.lin = nn.Linear(in_channels, effective_out_channels)
            print(f"Placeholder GATConv: in={in_channels}, out_channels_per_head={out_channels}, heads={heads}, effective_out={effective_out_channels}")
        def forward(self, x, edge_index):
            return self.lin(x) # Simplified placeholder

from .stm_gnn import STMGNNLayer # Our custom layer

class OriginalGNNModelWithGAT(nn.Module):
    """
    A hypothetical GNN model that uses GATConv layers.
    This is the model BEFORE replacing GATConv with STMGNNLayer.
    """
    def __init__(self, num_features, hidden_dim, num_classes, num_gat_layers=2, heads=8, dropout=0.1):
        super(OriginalGNNModelWithGAT, self).__init__()
        self.gat_layers = nn.ModuleList()

        current_dim = num_features
        for i in range(num_gat_layers):
            is_last_gnn_layer = (i == num_gat_layers - 1)
            # For GAT, output dim is heads * hidden_dim if concat=True
            # Last layer often has concat=False or heads=1 to get final desired dim,
            # or is followed by a linear layer.
            if is_last_gnn_layer:
                # Example: Last GAT layer outputs num_classes directly or a final embedding dim
                # Let's assume it outputs hidden_dim and a final Linear layer maps to num_classes
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
                ) # Output: hidden_dim
                current_dim = hidden_dim
            else:
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
                ) # Output: hidden_dim * heads
                current_dim = hidden_dim * heads

        self.output_lin = nn.Linear(current_dim, num_classes) # Maps final GNN output to classes
        self.dropout_val = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) -1: # No activation/dropout after last GAT's output before final linear
                 x = F.elu(x)
                 x = F.dropout(x, p=self.dropout_val, training=self.training)

        # Here, x is node embeddings. For graph classification, pool them.
        # For simplicity, assume node-level task or pooling is handled outside.
        # Or, apply pooling here if it's a graph classification model
        # x_graph = torch.mean(x, dim=0) # Example graph pooling
        # output = self.output_lin(x_graph)

        output = self.output_lin(x) # If node classification, or if output_lin handles pooled features
        return output


class ModelWithSTMGNNLayer(nn.Module):
    """
    The GNN model AFTER replacing a block of GATConv layers
    with a block of STMGNNLayer.
    The prompt implies replacing a "blok GATConv" (GATConv block)
    with "STMGNNLayer(hidden_dim=256, heads=8)".
    This suggests STMGNNLayer might be used repeatedly, similar to GATConv.
    """
    # Defaults hidden_dim_stm=256, heads_stm=8, num_stm_layers=5, dropout_stm=0.1, global_memory_dim_stm=128
    # as per various parts of AUROC spec for STM-GNN components
    def __init__(self, num_features, hidden_dim_stm=256, num_classes=None,
                 num_stm_layers=5, heads_stm=8, dropout_stm=0.1,
                 global_memory_dim_stm=128, num_memory_slots_stm=10,
                 time_embedding_dim_stm=None):
        super(ModelWithSTMGNNLayer, self).__init__()

        self.num_features = num_features
        self.hidden_dim_stm = hidden_dim_stm # As specified: hidden_dim=256 for STMGNNLayer

        # Input embedding to match STMGNNLayer's expected input dimension (hidden_dim_stm)
        self.input_embed = nn.Linear(num_features, hidden_dim_stm)
        current_dim = hidden_dim_stm

        self.stm_layers = nn.ModuleList()
        for _ in range(num_stm_layers): # Spec: 5 layers
            self.stm_layers.append(STMGNNLayer(
                in_channels=current_dim,
                out_channels=current_dim, # STMGNNLayer typically keeps dim same or maps in->out
                time_channels=time_embedding_dim_stm,
                memory_channels=global_memory_dim_stm,
                num_heads=heads_stm, # Spec: heads=8
                dropout=dropout_stm, # Spec: dropout=0.1
                concat=True # Explicitly set, though it's the default
            ))
            # current_dim remains hidden_dim_stm for subsequent STMGNN layers

        # Global Memory (if used and managed by this model, not just passed in)
        self.global_memory = None
        if global_memory_dim_stm and num_memory_slots_stm:
             self.global_memory = nn.Parameter(torch.randn(num_memory_slots_stm, global_memory_dim_stm))


        # Output layer, depends on the task (e.g., graph classification, node classification)
        if num_classes is not None:
            # Example for graph classification using mean pooling of node features
            self.graph_pool = lambda x_nodes: torch.mean(x_nodes, dim=0) # Pool node features for graph rep

            # Final representation might also include global memory
            final_repr_dim = current_dim # From node features
            if self.global_memory is not None:
                # Simple way: mean pool memory slots and concatenate
                final_repr_dim += global_memory_dim_stm

            self.output_classifier = nn.Linear(final_repr_dim, num_classes)


    def forward(self, x_initial_nodes, edge_index, time_embedding=None):
        """
        Forward pass for a single graph snapshot.
        If handling sequences of snapshots (like the main STMGNN class), this model
        would be a component within that larger temporal architecture.
        This example assumes it processes one graph at a time.
        """
        # 1. Embed initial node features
        node_repr = F.relu(self.input_embed(x_initial_nodes))

        # 2. Pass through STMGNN layers
        for layer in self.stm_layers:
            node_repr = layer(
                x=node_repr,
                edge_index=edge_index,
                time_embedding=time_embedding, # Pass if available
                global_memory=self.global_memory # Pass if this model manages it
            )
            # Note: STMGNNLayer's sketch includes a residual connection and LayerNorm.

        # 3. Prediction (example for graph classification)
        if hasattr(self, 'output_classifier'):
            graph_node_pooled_repr = self.graph_pool(node_repr) # (hidden_dim_stm)

            final_combined_repr = graph_node_pooled_repr
            if self.global_memory is not None:
                # Pool memory slots and concatenate
                pooled_memory_repr = torch.mean(self.global_memory, dim=0) # (global_memory_dim_stm)
                final_combined_repr = torch.cat([graph_node_pooled_repr, pooled_memory_repr], dim=-1)

            logits = self.output_classifier(final_combined_repr)
            return logits

        return node_repr # Return node embeddings if no classification head here


if __name__ == '__main__':
    print("--- Comparing OriginalGNNModelWithGAT vs ModelWithSTMGNNLayer ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy data for a single graph
    num_nodes_ex = 10
    num_features_ex = 16
    edge_index_ex = torch.randint(0, num_nodes_ex, (2, 20), device=device, dtype=torch.long)
    x_nodes_ex = torch.randn(num_nodes_ex, num_features_ex, device=device)
    num_classes_ex = 3 # Example for 3-class classification

    # --- Original Model with GATConv ---
    print("\n--- Original GNN Model (with GATConv) ---")
    original_model = OriginalGNNModelWithGAT(
        num_features=num_features_ex,
        hidden_dim=64, # GAT hidden dim
        num_classes=num_classes_ex,
        num_gat_layers=2,
        heads=4, # GAT heads
        dropout=0.1
    ).to(device)
    print(original_model)
    try:
        # Assuming model outputs node-level predictions or has internal pooling for graph
        # For this test, let's assume it needs reshaping or pooling for graph classification.
        # If it's node classification, output shape will be (num_nodes, num_classes)
        original_output_nodes = original_model(x_nodes_ex, edge_index_ex)
        print(f"Original model (node output) shape: {original_output_nodes.shape}")
        # If graph classification, apply pooling:
        original_output_graph = torch.mean(original_output_nodes, dim=0) # Pool for graph-level
        print(f"Original model (pooled graph output for classif.) shape: {original_output_graph.shape}")
        # If output_lin in OriginalGNN expects pooled features, this is wrong.
        # The current OriginalGNNModelWithGAT applies output_lin to node features directly.
    except Exception as e:
        print(f"Error running OriginalGNNModelWithGAT: {e}")


    # --- Model with STMGNNLayer ---
    # As per spec: STMGNNLayer(hidden_dim=256, heads=8)
    # 5 layers, dropout=0.1, global_memory=128D
    print("\n--- Model with STMGNNLayer ---")
    stm_based_model = ModelWithSTMGNNLayer(
        num_features=num_features_ex,
        hidden_dim_stm=64, # For quick test, spec says 256
        num_classes=num_classes_ex, # For graph classification
        num_stm_layers=2,      # For quick test, spec says 5
        heads_stm=2,           # For quick test, spec says 8
        dropout_stm=0.1,
        global_memory_dim_stm=32, # For quick test, spec says 128
        num_memory_slots_stm=5    # Example
    ).to(device)
    print(stm_based_model)
    try:
        # This model's forward pass is designed for graph classification
        stm_model_output = stm_based_model(x_nodes_ex, edge_index_ex)
        if stm_model_output is not None:
            print(f"STM-based model output shape: {stm_model_output.shape}") # (num_classes)
    except Exception as e:
        print(f"Error running ModelWithSTMGNNLayer: {e}")
        if not PYG_AVAILABLE:
             print("This might be due to STMGNNLayer's placeholder nature without PyTorch Geometric.")

    print("\nComparison complete. Note that STMGNNLayer is a conceptual sketch.")
