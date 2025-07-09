import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
# AddLaplacianEigenvectorPE is used in data_utils, not directly here usually

class GPSLayer(nn.Module):
    """
    A single GraphGPS layer, which typically combines a message passing GNN layer,
    a local Transformer-style attention (e.g., MHA on nodes), and positional encodings.
    This is a simplified placeholder. A full GRIT/SAN implementation would be more complex.
    """
    def __init__(self, dim_h, num_heads=4, dropout=0.1, attention_dropout=0.1, local_gnn_type='GCNConv', activation_fn_str='relu'):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads

        if activation_fn_str == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn_str == 'gelu':
            self.activation_fn = nn.GELU()
        elif activation_fn_str == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation_fn_str: {activation_fn_str}")

        # Local GNN part (Message Passing)
        if local_gnn_type == 'GCNConv':
            self.local_gnn = pyg_nn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GINConv':
            gin_nn = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_h))
            self.local_gnn = pyg_nn.GINConv(gin_nn)
        elif local_gnn_type == 'GATConv': # Default for predictors

            self.local_gnn = pyg_nn.GATConv(dim_h, dim_h, heads=num_heads, concat=False, dropout=attention_dropout)
        else:
            raise ValueError(f"Unsupported local_gnn_type: {local_gnn_type}")

        # Global Attention part (Self-Attention on nodes)
        self.self_attn = nn.MultiheadAttention(dim_h, num_heads, dropout=attention_dropout, batch_first=True)

        self.norm1_local = nn.LayerNorm(dim_h)
        self.norm1_attn = nn.LayerNorm(dim_h) # Changed from norm1_local to norm1_attn

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2),
            self.activation_fn, # Use the chosen activation function
            nn.Dropout(dropout),
            nn.Linear(dim_h * 2, dim_h)
        )
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_res_local = x
        x_local_out = self.local_gnn(x, edge_index)
        x_local_out = self.dropout_local(x_local_out)
        x = x_res_local + x_local_out
        x = self.norm1_local(x)

        x_res_attn = x
        # MultiheadAttention expects shape (N, L, E) or (L, N, E) if batch_first=False
        # Here, num_nodes is L (sequence length), N is batch_size (1 for a single graph)
        x_attn_input = x.unsqueeze(0) # Shape: [1, num_nodes, dim_h]
        x_attn, _ = self.self_attn(x_attn_input, x_attn_input, x_attn_input)
        x_attn = x_attn.squeeze(0) # Shape: [num_nodes, dim_h]
        x_attn = self.dropout_attn(x_attn)
        x = x_res_attn + x_attn
        x = self.norm1_attn(x) # Corrected norm application

        x_res_ffn = x
        x_ffn_out = self.ffn(x)
        x_ffn_out = self.dropout_ffn(x_ffn_out)
        x = x_res_ffn + x_ffn_out

        x = self.norm2(x)

        return x

class GraphGPS(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out_graph_level, num_layers, num_heads, # dim_out_graph_level for graph-level tasks
                 lap_pe_dim, sign_pe_dim, # These are the dimensions of PE features *provided in the Data object*
                 activation_fn_str='relu', # Added activation function string
                 dropout=0.1, local_gnn_type='GCNConv', pool_type='mean'):
        super().__init__()
        self.dim_h = dim_h
        self.lap_pe_dim = lap_pe_dim
        self.sign_pe_dim = sign_pe_dim

        # The atom_encoder's input dimension depends on raw features + PEs that will be concatenated
        effective_atom_encoder_dim_in = dim_in
        if self.lap_pe_dim > 0:
            effective_atom_encoder_dim_in += self.lap_pe_dim
        if self.sign_pe_dim > 0: # Placeholder for SignNet feature dimension
            effective_atom_encoder_dim_in += self.sign_pe_dim

        self.atom_encoder = nn.Linear(effective_atom_encoder_dim_in, dim_h)

        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gps_layers.append(GPSLayer(dim_h, num_heads, dropout, dropout, local_gnn_type, activation_fn_str=activation_fn_str))

        self.pool_type = pool_type
        if self.pool_type: # Only define pool and graph-level head if pooling is specified
            if pool_type == 'mean':
                self.pool = pyg_nn.global_mean_pool
            elif pool_type == 'add':
                self.pool = pyg_nn.global_add_pool
            elif pool_type == 'max':
                self.pool = pyg_nn.global_max_pool
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}")
            self.graph_level_output_head = nn.Linear(dim_h, dim_out_graph_level)
        else: # No pooling, implies node-level tasks handled by derived classes
            self.pool = None
            self.graph_level_output_head = None


    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        pe_features_list = []
        if self.lap_pe_dim > 0 and hasattr(data, 'lap_pe') and data.lap_pe is not None:
            # Ensure lap_pe has the expected dimension self.lap_pe_dim
            if data.lap_pe.shape[1] != self.lap_pe_dim:
                 raise ValueError(f"data.lap_pe dim {data.lap_pe.shape[1]} != model lap_pe_dim {self.lap_pe_dim}")
            pe_features_list.append(data.lap_pe)

        if self.sign_pe_dim > 0 and hasattr(data, 'sign_pe') and data.sign_pe is not None:
            if data.sign_pe.shape[1] != self.sign_pe_dim:
                 raise ValueError(f"data.sign_pe dim {data.sign_pe.shape[1]} != model sign_pe_dim {self.sign_pe_dim}")
            pe_features_list.append(data.sign_pe)

        if pe_features_list:
            x = torch.cat([x] + pe_features_list, dim=-1)

        x = self.atom_encoder(x) # Now x has dim_h

        for layer in self.gps_layers:
            x = layer(x, edge_index) # Node embeddings after GPS layers

        # If pooling is defined (for graph-level tasks), apply it and the graph-level head
        if self.pool is not None and self.graph_level_output_head is not None:
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_emb = self.pool(x, batch)
            out = self.graph_level_output_head(graph_emb)
            return out # Graph-level output
        else:
            # For node-level tasks, derived classes will use these node embeddings
            return x # Node embeddings, shape [num_nodes, dim_h]



class MortalityPredictor(GraphGPS):
    def __init__(self, dim_in, dim_h, num_layers, num_heads, lap_pe_dim, sign_pe_dim, dropout=0.1, activation_fn_str='relu'):
        # dim_out_graph_level for GraphGPS base is not used as pool_type is None
        super().__init__(dim_in=dim_in, dim_h=dim_h, dim_out_graph_level=1,
                         num_layers=num_layers, num_heads=num_heads,
                         lap_pe_dim=lap_pe_dim, sign_pe_dim=sign_pe_dim,
                         activation_fn_str=activation_fn_str, # Pass activation
                         dropout=dropout, local_gnn_type='GATConv', pool_type=None) # No global pooling
        self.node_level_output_head = nn.Linear(dim_h, 1) # Specific head for this task

    def forward(self, data: Data):
        node_embeddings = super().forward(data) # Get [num_nodes, dim_h]
        out = self.node_level_output_head(node_embeddings) # Apply specific head: [num_nodes, 1]
        return out

class LoSPredictor(GraphGPS):
    def __init__(self, dim_in, dim_h, num_layers, num_heads, lap_pe_dim, sign_pe_dim, dropout=0.1, activation_fn_str='relu'):
        super().__init__(dim_in=dim_in, dim_h=dim_h, dim_out_graph_level=1,
                         num_layers=num_layers, num_heads=num_heads,
                         lap_pe_dim=lap_pe_dim, sign_pe_dim=sign_pe_dim,
                         activation_fn_str=activation_fn_str, # Pass activation
                         dropout=dropout, local_gnn_type='GATConv', pool_type=None) # No global pooling
        self.node_level_output_head = nn.Linear(dim_h, 1) # Specific head for this task

    def forward(self, data: Data):
        node_embeddings = super().forward(data) # Get [num_nodes, dim_h]
        out = self.node_level_output_head(node_embeddings) # Apply specific head: [num_nodes, 1]
        return out

if __name__ == '__main__':
    print("--- Testing models.py (Revised for Node-Level Predictions v2) ---")

    DIM_RAW_FEATURES = 75  # Features from ColumnTransformer in data_utils
    LAP_PE_K_DIM = 8       # Dimension of LapPE features from data_utils
    SIGN_PE_K_DIM = 0      # Dimension of SignNet features (0 for now)

    DIM_HIDDEN = 128
    NUM_GPS_LAYERS = 3
    NUM_ATTN_HEADS = 4
    DROPOUT_RATE = 0.1

    print(f"Model Parameters:\n  Raw Features Dim (dim_in for model): {DIM_RAW_FEATURES}")
    print(f"  LapPE Dim (lap_pe_dim for model): {LAP_PE_K_DIM}")
    print(f"  SignNet Dim (sign_pe_dim for model): {SIGN_PE_K_DIM}")
    print(f"  Effective Input Dim to Atom Encoder (dim_raw + lap_pe_dim + sign_pe_dim): {DIM_RAW_FEATURES + LAP_PE_K_DIM + SIGN_PE_K_DIM}")
    print(f"  Hidden Dim: {DIM_HIDDEN}\n  GPS Layers: {NUM_GPS_LAYERS}\n  Attn Heads: {NUM_ATTN_HEADS}\n  Dropout: {DROPOUT_RATE}")

    num_nodes = 20
    num_edges = 50

    dummy_x_raw = torch.randn(num_nodes, DIM_RAW_FEATURES)
    dummy_lap_pe = torch.randn(num_nodes, LAP_PE_K_DIM) if LAP_PE_K_DIM > 0 else None

    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    dummy_data = Data(x=dummy_x_raw, edge_index=dummy_edge_index)

    if dummy_lap_pe is not None:
        dummy_data.lap_pe = dummy_lap_pe
    # dummy_data.sign_pe will be None if SIGN_PE_K_DIM is 0

    print(f"\nDummy input graph data structure:\n{dummy_data}")
    print(f"dummy_data.x shape: {dummy_data.x.shape}")
    if hasattr(dummy_data, 'lap_pe') and dummy_data.lap_pe is not None:
        print(f"dummy_data.lap_pe shape: {dummy_data.lap_pe.shape}")


    print("\n--- Testing MortalityPredictor ---")
    # dim_in to MortalityPredictor is the dimension of raw features (data.x)
    # lap_pe_dim is the dimension of data.lap_pe (if present)
    # sign_pe_dim is the dimension of data.sign_pe (if present)
    mortality_model = MortalityPredictor(
        dim_in=DIM_RAW_FEATURES,
        dim_h=DIM_HIDDEN, num_layers=NUM_GPS_LAYERS, num_heads=NUM_ATTN_HEADS,
        lap_pe_dim=LAP_PE_K_DIM if dummy_lap_pe is not None else 0,
        sign_pe_dim=SIGN_PE_K_DIM, # SIGN_PE_K_DIM is 0
        dropout=DROPOUT_RATE

    )
    print(f"Mortality model structure:\n{mortality_model}")
    try:
        mortality_output = mortality_model(dummy_data)
        print(f"Mortality model output shape: {mortality_output.shape}")
        assert mortality_output.shape == (num_nodes, 1)
        print(f"Mortality model output sample (first 5):\n{mortality_output[:5]}")

    except Exception as e:
        print(f"Error during MortalityPredictor forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing LoSPredictor ---")
    los_model = LoSPredictor(
        dim_in=DIM_RAW_FEATURES,
        dim_h=DIM_HIDDEN, num_layers=NUM_GPS_LAYERS, num_heads=NUM_ATTN_HEADS,
        lap_pe_dim=LAP_PE_K_DIM if dummy_lap_pe is not None else 0,
        sign_pe_dim=SIGN_PE_K_DIM,
        dropout=DROPOUT_RATE

    )
    print(f"LoS model structure:\n{los_model}")
    try:
        los_output = los_model(dummy_data)
        print(f"LoS model output shape: {los_output.shape}")
        assert los_output.shape == (num_nodes, 1)
        print(f"LoS model output sample (first 5):\n{los_output[:5]}")

    except Exception as e:
        print(f"Error during LoSPredictor forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Note on Positional Encodings (LapPE & SignNet) ---")
    print("PEs (if lap_pe_dim > 0 or sign_pe_dim > 0) are now expected to be concatenated to data.x ")
    print("BEFORE the atom_encoder in the GraphGPS base class.")
    print("The `dim_in` to GraphGPS constructor should be the dimension of the raw features from data_utils,")
    print("and `lap_pe_dim`, `sign_pe_dim` should be the dimensions of these PEs as they appear in the Data object.")
    print("The `atom_encoder` inside GraphGPS will then handle the combined dimension.")

    print("\nmodels.py revised structure and tests complete.")
