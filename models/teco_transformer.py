import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformers.
    Adds positional information to input embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model) - for batch compatibility
        self.register_buffer('pe', pe) # Not a parameter, but part of state

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings, shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Embeddings with added positional encoding, same shape as x.
        """
        # x.size(1) is the sequence length (L)
        # self.pe is (1, max_len, d_model). We need (1, L, d_model)
        # Slicing self.pe ensures it matches the input sequence length L.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TECOTransformerModel(nn.Module):
    """
    TECO-Transformer (Conceptual - based on standard Transformer Encoder).
    "4-layer encoder, d_model = 512"

    This model processes sequences of features (e.g., time-series EHR data).
    """
    # Defaults d_model=512, num_encoder_layers=4 as per AUROC spec
    # nhead=8, dim_feedforward=2048 are common for d_model=512
    def __init__(self, input_feature_dim, d_model=512, num_encoder_layers=4,
                 nhead=8, dim_feedforward=2048, dropout=0.1, num_classes=None,
                 max_seq_len=500):
        """
        Args:
            input_feature_dim (int): Dimensionality of input features at each time step.
            d_model (int): The number of expected features in the encoder inputs. Default: 512.
            num_encoder_layers (int): Number of sub-encoder-layers in the encoder. Default: 4.
            nhead (int): Number of heads in the multiheadattention models. Default: 8.
            dim_feedforward (int): Dimension of the feedforward network model. Default: 2048.
            dropout (float): Dropout value. Default: 0.1.
            num_classes (int, optional): Number of output classes for classification.
                                         If None, model returns sequence embeddings.
            max_seq_len (int): Maximum sequence length for positional encoding. Default: 500.
        """
        super(TECOTransformerModel, self).__init__()
        self.d_model = d_model

        # 1. Input Embedding/Projection
        # Project input features to d_model if they are not already that dimension.
        self.input_projection = nn.Linear(input_feature_dim, self.d_model) # Use self.d_model

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=max_seq_len) # Use self.d_model

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, # Use self.d_model
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Assumes input shape (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Output Layer (for classification)
        self.num_classes = num_classes
        if self.num_classes is not None:
            # Use the embedding of the [CLS] token or mean pool sequence output
            # For simplicity, let's assume mean pooling of the output sequence
            self.output_classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers
        # (Often beneficial for Transformer models)
        for name, p in self.named_parameters():
            if p.dim() > 1 and ("transformer_encoder" in name or "input_projection" in name or "output_classifier" in name):
                if "weight" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(self, src_sequences, src_padding_mask=None):
        """
        Args:
            src_sequences (Tensor): Source sequences.
                                    Shape: (batch_size, seq_len, input_feature_dim).
            src_padding_mask (BoolTensor, optional): Mask for padding tokens in src_sequences.
                                                    Shape: (batch_size, seq_len).
                                                    True for padded positions, False otherwise.
        Returns:
            Tensor: If num_classes is set, returns classification logits (batch_size, num_classes).
                    Otherwise, returns sequence embeddings (batch_size, seq_len, d_model).
        """
        # 1. Project input features to d_model
        # (batch_size, seq_len, input_feature_dim) -> (batch_size, seq_len, d_model)
        x = self.input_projection(src_sequences) * math.sqrt(self.d_model) # Scale embeddings

        # 2. Add positional encoding
        x = self.pos_encoder(x)

        # 3. Pass through Transformer Encoder
        # src_key_padding_mask should be True for padded items.
        # nn.TransformerEncoderLayer expects (seq_len, batch_size, d_model) if batch_first=False (default)
        # But we use batch_first=True, so (batch_size, seq_len, d_model) is fine.
        encoded_sequence = self.transformer_encoder(x, src_key_padding_mask=src_padding_mask)
        # Shape: (batch_size, seq_len, d_model)

        # 4. Classification (if applicable)
        if self.num_classes is not None:
            # Simple mean pooling over sequence dimension, ignoring padding
            if src_padding_mask is not None:
                # Invert mask for summing: True for valid tokens, False for padding
                valid_tokens_mask = ~src_padding_mask # (batch_size, seq_len)
                valid_tokens_mask = valid_tokens_mask.unsqueeze(-1).expand_as(encoded_sequence) # (B, S, D)

                sum_embeddings = (encoded_sequence * valid_tokens_mask).sum(dim=1) # (B, D)
                num_valid_tokens = valid_tokens_mask.sum(dim=1) # (B, D), sum over S
                num_valid_tokens = torch.clamp(num_valid_tokens[:,0], min=1) # (B,), take one feature dim count

                pooled_output = sum_embeddings / num_valid_tokens.unsqueeze(-1) # (B,D) / (B,1)
            else:
                pooled_output = encoded_sequence.mean(dim=1) # (batch_size, d_model)

            logits = self.output_classifier(pooled_output) # (batch_size, num_classes)
            return logits
        else:
            return encoded_sequence


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("--- TECO-Transformer (Conceptual) Example ---")

    # Configuration
    batch_size_ex = 4
    seq_len_ex = 50    # Length of input sequences (e.g., time steps)
    input_feats_ex = 20 # Number of features at each time step

    d_model_ex = 128 # For faster test, spec says 512
    n_layers_ex = 2  # For faster test, spec says 4
    n_heads_ex = 4   # Number of attention heads
    n_classes_ex = 3 # Example for 3-class classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize TECO-Transformer model
    teco_model = TECOTransformerModel(
        input_feature_dim=input_feats_ex,
        d_model=d_model_ex,
        num_encoder_layers=n_layers_ex,
        nhead=n_heads_ex,
        num_classes=n_classes_ex,
        max_seq_len=seq_len_ex + 10 # Ensure max_len is sufficient
    ).to(device)
    print(f"\nTECO-Transformer Model structure:\n{teco_model}")

    # 2. Create dummy input data (sequences)
    # (batch_size, seq_len, input_feature_dim)
    dummy_sequences = torch.randn(batch_size_ex, seq_len_ex, input_feats_ex, device=device)

    # Create a dummy padding mask (e.g., last 10 elements of 2nd sequence are padding)
    dummy_padding_mask = torch.zeros(batch_size_ex, seq_len_ex, dtype=torch.bool, device=device)
    if batch_size_ex > 1 and seq_len_ex > 10:
        dummy_padding_mask[1, -10:] = True

    print(f"\nInput sequences shape: {dummy_sequences.shape}")
    print(f"Padding mask shape: {dummy_padding_mask.shape}, example sum: {dummy_padding_mask[1].sum()}")


    # 3. Forward pass
    try:
        logits_output = teco_model(dummy_sequences, src_padding_mask=dummy_padding_mask)
        print(f"\nModel output (logits) shape: {logits_output.shape}") # (batch_size, num_classes)

        # Conceptual: Training step
        # dummy_targets = torch.randint(0, n_classes_ex, (batch_size_ex,), device=device)
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(logits_output, dummy_targets)
        # print(f"Conceptual loss: {loss.item()}")
        # loss.backward() # To check if gradients flow
        # print("Conceptual backward pass successful.")

    except Exception as e:
        print(f"\nError during TECO-Transformer forward pass: {e}")

    # Example: Get sequence embeddings (if num_classes=None)
    teco_embedder = TECOTransformerModel(
        input_feature_dim=input_feats_ex,
        d_model=d_model_ex,
        num_encoder_layers=n_layers_ex,
        nhead=n_heads_ex,
        num_classes=None # Output embeddings
    ).to(device)
    sequence_embeddings = teco_embedder(dummy_sequences, src_padding_mask=dummy_padding_mask)
    print(f"\nModel output (sequence embeddings) shape: {sequence_embeddings.shape}") # (batch, seq, d_model)

    print("\n--- TECO-Transformer Example Finished ---")
