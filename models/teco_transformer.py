import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# It's good practice to get a logger instance for the module
# Assuming logger is configured appropriately at the application level
logger = logging.getLogger(__name__)


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
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model) - for batch compatibility
        self.register_buffer('pe', pe)  # Not a parameter, but part of state

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
        # Add check for x.size(1) > self.pe.shape[1] (max_len)
        if x.size(1) > self.pe.shape[1]:
            raise ValueError(
                f"Input sequence length ({x.size(1)}) exceeds PositionalEncoding max_len ({self.pe.shape[1]})"
            )
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
                 max_seq_len=500):  # max_seq_len for PositionalEncoding
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
        self.max_seq_len_config = max_seq_len  # Store for clarity if needed for debug

        # 1. Input Embedding/Projection
        # Project input features to d_model if they are not already that dimension.
        self.input_projection = nn.Linear(input_feature_dim, self.d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=self.max_seq_len_config)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Assumes input shape (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Output Layer (for classification)
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.output_classifier = nn.Linear(self.d_model, self.num_classes)  # Use self.d_model and self.num_classes

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers
        for name, p in self.named_parameters():
            if p.dim() > 1 and (
                    "transformer_encoder" in name or "input_projection" in name or "output_classifier" in name):
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
        # Initial assertions for src_sequences
        assert isinstance(src_sequences, torch.Tensor), f"src_sequences must be a Tensor. Got {type(src_sequences)}"
        assert src_sequences.dim() == 3, f"src_sequences must be 3D (batch, seq, feature_input_dim). Got {src_sequences.dim()}D, shape {src_sequences.shape}"
        # Actual sequence length from input
        current_seq_len = src_sequences.shape[1]
        if current_seq_len > self.max_seq_len_config:
            logger.warning(
                f"Input sequence length ({current_seq_len}) for a batch "
                f"exceeds model's configured max_seq_len for PositionalEncoding ({self.max_seq_len_config}). "
                f"This will cause an error in PositionalEncoding. Ensure data loader truncates/pads sequences appropriately "
                f"or reconfigure model with larger max_seq_len if dynamic sequence lengths are expected up to this new max."
            )
            # The PositionalEncoding.forward will raise a more specific error if this happens.

        # 1. Project input features to d_model
        x = self.input_projection(src_sequences) * math.sqrt(self.d_model)  # Scale embeddings

        # 2. Add positional encoding
        x = self.pos_encoder(x)  # x shape: (batch_size, current_seq_len, d_model)

        # --- Start Debugging/Assertion Block for TransformerEncoder input ---
        assert isinstance(x,
                          torch.Tensor), f"Input 'x' to TransformerEncoder (after pos_encoder) must be a Tensor. Got {type(x)}"
        assert x.dim() == 3, f"Input 'x' to TransformerEncoder must be 3D (batch, seq, d_model). Got {x.dim()}D, shape {x.shape}"

        # Batch size check (can be done once, e.g. on src_sequences.shape[0])
        batch_size = x.shape[0]
        if not (batch_size > 0):
            logger.warning(
                f"Input 'x' batch size is {batch_size}. Expected > 0. This might be an empty batch if drop_last=False and dataset size is small.")
        # Allowing empty batch to proceed to transformer_encoder if PyTorch handles it, error will be caught by try-except.
        # assert batch_size > 0, "Input 'x' batch size must be > 0."

        assert current_seq_len > 0, f"Input 'x' sequence length must be > 0. Got {current_seq_len}"  # current_seq_len from src_sequences
        assert x.shape[
                   1] == current_seq_len, f"Sequence length of x ({x.shape[1]}) after pos_encoder should match current_seq_len ({current_seq_len})."
        assert x.shape[
                   2] == self.d_model, f"Input 'x' feature dim after projection must be d_model ({self.d_model}). Got {x.shape[2]}"

        if src_padding_mask is not None:
            assert isinstance(src_padding_mask,
                              torch.Tensor), f"src_padding_mask must be a Tensor. Got {type(src_padding_mask)}"
            assert src_padding_mask.dim() == 2, f"src_padding_mask must be 2D (batch, seq). Got {src_padding_mask.dim()}D, shape {src_padding_mask.shape}"
            assert src_padding_mask.shape[
                       0] == batch_size, f"src_padding_mask batch size ({src_padding_mask.shape[0]}) must match input 'x' batch size ({batch_size})."
            assert src_padding_mask.shape[
                       1] == current_seq_len, f"src_padding_mask sequence length ({src_padding_mask.shape[1]}) must match input 'x' sequence length ({current_seq_len})."
            assert src_padding_mask.dtype == torch.bool, f"src_padding_mask dtype must be torch.bool. Got {src_padding_mask.dtype}"
        # --- End Debugging/Assertion Block ---

        # 3. Pass through Transformer Encoder
        try:
            encoded_sequence = self.transformer_encoder(x, src_key_padding_mask=src_padding_mask)
        except Exception as e:
            # Log detailed info before re-raising
            logger.error(f"Error during self.transformer_encoder(x, src_key_padding_mask=src_padding_mask) call.")
            logger.error(f"  x properties: shape={x.shape}, dtype={x.dtype}, device={x.device}")
            if src_padding_mask is not None:
                logger.error(
                    f"  src_padding_mask properties: shape={src_padding_mask.shape}, dtype={src_padding_mask.dtype}, device={src_padding_mask.device}")
                if src_padding_mask.numel() > 0 and src_padding_mask.shape[
                    0] > 0:  # Check if mask has elements and at least one batch item
                    sample_mask_slice = src_padding_mask[0, :min(5, src_padding_mask.shape[1])]
                    logger.error(f"  src_padding_mask sample (first batch item, up to 5 elements): {sample_mask_slice}")
                else:
                    logger.error(f"  src_padding_mask is empty or has no batch items to sample.")
            else:
                logger.error(f"  src_padding_mask: None")
            # Re-raise the caught exception to allow higher-level error handling (e.g., in train.py)
            raise e
        # encoded_sequence shape: (batch_size, current_seq_len, d_model)

        # 4. Classification (if applicable)
        if self.num_classes is not None:
            # Simple mean pooling over sequence dimension, ignoring padding
            if src_padding_mask is not None:
                inverted_padding_mask = ~src_padding_mask  # True for valid tokens
                valid_tokens_mask_expanded = inverted_padding_mask.unsqueeze(-1).float()

                masked_encoded_sequence = encoded_sequence * valid_tokens_mask_expanded
                sum_embeddings = masked_encoded_sequence.sum(dim=1)

                num_valid_tokens_per_sequence = inverted_padding_mask.sum(dim=1).float()
                num_valid_tokens_per_sequence = torch.clamp(num_valid_tokens_per_sequence,
                                                            min=1)  # Avoid division by zero

                pooled_output = sum_embeddings / num_valid_tokens_per_sequence.unsqueeze(-1)
            else:
                # No padding mask provided, assume all tokens are valid
                pooled_output = encoded_sequence.mean(dim=1)

            logits = self.output_classifier(pooled_output)
            return logits
        else:
            # If no classification head, return the full sequence embeddings
            return encoded_sequence


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("--- TECO-Transformer (Conceptual) Example ---")

    # Configuration
    batch_size_ex = 4
    seq_len_ex = 50  # Length of input sequences (e.g., time steps)
    input_feats_ex = 20  # Number of features at each time step

    d_model_ex = 128  # For faster test, spec says 512
    n_layers_ex = 2  # For faster test, spec says 4
    n_heads_ex = 4  # Number of attention heads
    n_classes_ex = 3  # Example for 3-class classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize TECO-Transformer model
    teco_model = TECOTransformerModel(
        input_feature_dim=input_feats_ex,
        d_model=d_model_ex,
        num_encoder_layers=n_layers_ex,
        nhead=n_heads_ex,
        num_classes=n_classes_ex,
        max_seq_len=seq_len_ex + 10  # Ensure max_len is sufficient
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
        print(f"\nModel output (logits) shape: {logits_output.shape}")  # (batch_size, num_classes)

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
        num_classes=None  # Output embeddings
    ).to(device)
    sequence_embeddings = teco_embedder(dummy_sequences, src_padding_mask=dummy_padding_mask)
    print(f"\nModel output (sequence embeddings) shape: {sequence_embeddings.shape}")  # (batch, seq, d_model)

    print("\n--- TECO-Transformer Example Finished ---")
