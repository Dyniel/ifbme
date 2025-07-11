# models/hetero_temporal_gnn.py

"""
Defines a Heterogeneous Temporal Graph Neural Network model.
This could be a TGN-like model with GAT or GraphTransformer layers
for processing heterogeneous patient graphs over time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
# For TGN-like components, if we go that route:
# from torch_geometric.nn.models.tgn import TGNMemory, LastNeighborLoader # Or similar temporal components

# Placeholder for actual configuration that will come from YAML
# Example:
# num_global_vital_nodes = 100 # Size of vital_to_id mapper
# num_global_diag_nodes = 1000
# num_global_med_nodes = 500
# num_global_proc_nodes = 300
# time_slice_input_feature_dim = 16 (time_embed) + num_vital_features_aggregated
# embedding_dim_concepts = 64
# gnn_hidden_dim = 128
# num_gnn_layers = 2
# num_gat_heads = 4
# out_dim = 1 # For binary classification

class HeteroTemporalGNN(nn.Module):
    def __init__(self, data_schema, # For node_types, edge_types
                 num_nodes_dict, # Dict: {'vital': num_global_vitals, 'diagnosis': ...}
                 timeslice_feat_dim, # Input feature dim for 'timeslice' nodes
                 concept_embedding_dim, # Embedding dim for V, D, M, P nodes
                 gnn_hidden_dim,
                 gnn_output_dim, # Output dim from GNN layers before final FC
                 num_gnn_layers,
                 num_gat_heads,
                 output_classes=1, # For binary classification
                 dropout_rate=0.3):
        super().__init__()

        self.node_types = data_schema.NODE_TYPES
        self.edge_types = data_schema.EDGE_TYPES # Canonical edge types

        self.num_nodes_dict = num_nodes_dict
        self.concept_embedding_dim = concept_embedding_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_output_dim = gnn_output_dim # Output of GNN layers
        self.num_gnn_layers = num_gnn_layers
        self.output_classes = output_classes

        # --- 1. Node Embeddings / Initial Projections ---
        self.embeddings = nn.ModuleDict()
        self.initial_projections = nn.ModuleDict()

        # Embeddings for concept nodes (V, D, M, P)
        # Only create embeddings for node types that are active (not commented out in schema)
        # and have a non-zero number of unique items in their mappers.
        active_concept_node_types = [nt for nt in ['vital', 'diagnosis', 'medication', 'procedure'] if nt in self.node_types]

        for node_type in active_concept_node_types:
            if node_type in self.num_nodes_dict and self.num_nodes_dict[node_type] > 0:
                self.embeddings[node_type] = nn.Embedding(
                    num_embeddings=self.num_nodes_dict[node_type],
                    embedding_dim=self.concept_embedding_dim
                )
                # Project to gnn_hidden_dim if different
                self.initial_projections[node_type] = nn.Linear(
                    self.concept_embedding_dim, self.gnn_hidden_dim
                )
            else: # Should not happen if mappers are correct
                 print(f"Warning: num_nodes for {node_type} is 0 or not in num_nodes_dict.")


        # Initial projection for 'timeslice' node features
        if 'timeslice' in self.node_types: # timeslice nodes don't use Embedding, they have .x
            self.initial_projections['timeslice'] = nn.Linear(
                timeslice_feat_dim, self.gnn_hidden_dim
            )

        # --- 2. Heterogeneous GNN Layers (e.g., GAT based) ---
        self.convs = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                # Example: GATConv for all edge types
                # Input dimension for GATConv is self.gnn_hidden_dim (after initial projection)
                # Output dimension is also self.gnn_hidden_dim for intermediate layers
                # For the last layer, it could be self.gnn_output_dim
                # GATConv takes (-1, -1) for heterogeneous graphs if node features are of different sizes before projection
                # but after initial_projections, all node features going into HeteroConv will be gnn_hidden_dim

                # src_type, rel, dst_type = edge_type
                # For PyG's HeteroConv, keys are strings like 'paper__to__author' not tuples.
                # We might need to adapt how edge_types are stored or used here.
                # For now, let's assume we can construct the GATConv.
                # This part requires careful handling of HeteroConv structure.

                # Simple GAT for illustration, actual implementation might need specific handling for edge types
                # For HeteroConv, the GATConv input dim is the hidden_dim, output is hidden_dim / heads for multi-head attention
                # The HeteroConv layer will aggregate these.
                # We need one GATConv per edge_type.
                # Let's assume gnn_output_dim is per-head, so hidden_dim for layer output
                conv_dict[edge_type_to_str(edge_type)] = GATConv((-1,-1), # Input features from any source/dest type
                                                                self.gnn_hidden_dim // num_gat_heads, # out_channels per head
                                                                heads=num_gat_heads,
                                                                dropout=dropout_rate,
                                                                add_self_loops=False) # Self-loops often added in message passing

            self.convs.append(HeteroConv(conv_dict, aggr='sum')) # Or 'mean', 'max'

        # --- 3. Readout/Pooling Layer (Global Pooling) ---
        # This will be applied after the last GNN layer to get a graph-level embedding.
        # We might need a more sophisticated readout, e.g., attention-based.
        # For now, simple mean pooling over all node types, then concatenate or sum.

        # --- 4. Final Classifier ---
        # Input to this layer will be the graph embedding from readout.
        # Its dimension depends on the readout strategy and gnn_output_dim.
        # If we average gnn_output_dim vectors from each node type:
        # self.classifier_input_dim = self.gnn_hidden_dim (if last conv outputs gnn_hidden_dim)
        # For simplicity, let's assume the readout gives a vector of size gnn_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.gnn_hidden_dim // 2, self.output_classes)
        )

    def forward(self, data: HeteroData):
        x_dict = {} # To store node features for each type

        # Apply initial projections / embeddings
        for node_type in data.node_types:
            if node_type == 'timeslice':
                if data[node_type].num_nodes > 0:
                    x_dict[node_type] = self.initial_projections[node_type](data[node_type].x)
                else:
                    x_dict[node_type] = torch.empty((0, self.gnn_hidden_dim), device=data.device if hasattr(data,'device') else 'cpu')
            elif node_type in self.embeddings: # V, D, M, P
                if data[node_type].num_nodes > 0:
                    # Assuming node IDs for embeddings are simply torch.arange(data[node_type].num_nodes)
                    # if .x is not provided for these types.
                    # Or, if data[node_type].node_ids (global ids) are provided:
                    # node_indices = data[node_type].node_ids
                    # This requires careful handling in data preparation.
                    # For now, assume GNN is for a single graph, so node indices are 0 to N-1 for each type.
                    # The `num_nodes` in HeteroData for V/D/M/P should be set to max_global_id+1.
                    # The actual nodes present in a patient graph will be a subset, referenced by edge_index.
                    # So, we lookup embeddings for ALL global concept nodes.
                    all_concept_node_ids = torch.arange(self.num_nodes_dict[node_type], device=data.device if hasattr(data,'device') else 'cpu')
                    embedded_concepts = self.embeddings[node_type](all_concept_node_ids)
                    x_dict[node_type] = self.initial_projections[node_type](embedded_concepts)
                else: # Should not happen if num_nodes_dict[node_type]>0
                    x_dict[node_type] = torch.empty((0, self.gnn_hidden_dim), device=data.device if hasattr(data,'device') else 'cpu')
            # else: node_type not in projections or embeddings (should not happen if schema is covered)

        # Pass through GNN layers
        for conv_layer in self.convs:
            # HeteroConv expects edge_index_dict
            # Our data object has edge_index per edge type string key.
            # We need to ensure data.edge_index_dict() or similar is available or construct it.
            # Or, pass edge_index for each type explicitly if HeteroConv is iterated.
            # The HeteroConv layer handles passing the correct x_src, x_dst based on edge_type.
            x_dict = conv_layer(x_dict, data.edge_index_dict)
            # Apply activation and dropout (often done within GATConv or after HeteroConv)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            # x_dict = {key: F.dropout(x, p=self.dropout_rate, training=self.training) for key, x in x_dict.items()}


        # Readout / Global Pooling
        # Simple mean pooling of 'timeslice' node embeddings as graph representation
        # This is a simplification; a more robust readout would consider all node types
        # or use a dedicated global pooling layer (like global_mean_pool from PyG).
        if 'timeslice' in x_dict and x_dict['timeslice'].numel() > 0 :
            # Create a batch vector if not present (for global_add_pool, global_mean_pool)
            # If data comes from PyG DataLoader, data.batch_dict['timeslice'] should exist.
            batch_vec = data['timeslice'].batch if hasattr(data['timeslice'], 'batch') and data['timeslice'].batch is not None else torch.zeros(x_dict['timeslice'].size(0), dtype=torch.long, device=x_dict['timeslice'].device)

            # graph_embedding = global_mean_pool(x_dict['timeslice'], batch_vec) # Using PyG's global pooling
            # For simplicity if global_mean_pool is not directly usable or batching is complex:
            if x_dict['timeslice'].size(0) > 0:
                 graph_embedding = x_dict['timeslice'].mean(dim=0) # This is for a single graph
            else: # No timeslice nodes in this graph after processing
                 graph_embedding = torch.zeros(self.gnn_hidden_dim, device=x_dict['timeslice'].device if 'timeslice' in x_dict and x_dict['timeslice'].device else 'cpu')

        else: # No timeslice nodes at all (e.g. empty graph)
            # Fallback: zero vector for graph embedding
            # Determine device from other potential tensors or default to CPU
            device = 'cpu'
            if hasattr(data, 'device'): device = data.device
            elif any(isinstance(n, torch.Tensor) for n in x_dict.values()): # Get device from any tensor in x_dict
                for n_type in x_dict:
                    if isinstance(x_dict[n_type], torch.Tensor) and x_dict[n_type].numel() > 0:
                        device = x_dict[n_type].device
                        break
            graph_embedding = torch.zeros(self.gnn_hidden_dim, device=device)


        # Classifier
        logits = self.classifier(graph_embedding)

        if self.output_classes == 1: # For BCEWithLogitsLoss
            return logits.squeeze(-1)
        return logits # For CrossEntropyLoss (if output_classes > 1)

def edge_type_to_str(edge_type_tuple):
    """Converts ('src', 'rel', 'dst') to 'src__rel__dst' for HeteroConv keys."""
    return f"{edge_type_tuple[0]}__{edge_type_tuple[1]}__{edge_type_tuple[2]}"

if __name__ == '__main__':
    print("HeteroTemporalGNN structure outlined.")
    # TODO: Add example instantiation and forward pass with mock HeteroData
    # from data_utils.graph_schema import NODE_TYPES as schema_node_types, EDGE_TYPES as schema_edge_types
    # class MockSchema:
    #     NODE_TYPES = schema_node_types
    #     EDGE_TYPES = schema_edge_types

    # mock_schema = MockSchema()
    # mock_num_nodes = {
    #     'timeslice': 0, # num_nodes for timeslice is dynamic per graph
    #     'vital': 50, 'diagnosis': 200, 'medication': 100, 'procedure': 80
    # }
    # model = HeteroTemporalGNN(
    #     data_schema=mock_schema,
    #     num_nodes_dict=mock_num_nodes,
    #     timeslice_feat_dim=16+10, # 16 time embed + 10 vital features
    #     concept_embedding_dim=32,
    #     gnn_hidden_dim=64,
    #     gnn_output_dim=64, # Usually same as hidden_dim before final FC
    #     num_gnn_layers=2,
    #     num_gat_heads=4,
    #     output_classes=1
    # )
    # print(model)

    # Create a dummy HeteroData object
    # data = HeteroData()
    # data['timeslice'].x = torch.randn(10, 26) # 10 time slices, 16+10 features
    # data['timeslice'].num_nodes = 10
    # data['vital'].num_nodes = 50 # Global number of vital types
    # data['diagnosis'].num_nodes = 200
    # # ... and so on for other node types, even if no .x is assigned here

    # # Edges (example)
    # data['timeslice', 'temporally_precedes', 'timeslice'].edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    # data['timeslice', 'has_vital_measurement', 'vital'].edge_index = torch.tensor([[0,0,1],[0,1,0]], dtype=torch.long) # (ts_idx, vital_global_id)
    # data['vital', 'vital_measured_in', 'timeslice'].edge_index = torch.tensor([[0,1,0],[0,0,1]], dtype=torch.long) # (vital_global_id, ts_idx)

    # try:
    #     output = model(data)
    #     print("Dummy forward pass output shape:", output.shape)
    # except Exception as e:
    #     print("Error during dummy forward pass:", e)
    #     import traceback
    #     traceback.print_exc()

    pass
