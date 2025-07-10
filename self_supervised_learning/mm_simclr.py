import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning, as used in SimCLR.
    Maps a representation to a lower-dimensional space where contrastive loss is applied.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=1) # Normalize embeddings

class MMSimCLR(nn.Module):
    """
    Conceptual Multimodal SimCLR (MM-SimCLR) for EHR structured data and text notes.

    This model aims to learn aligned representations by contrasting paired
    structured EHR data and text note embeddings against other unpaired samples.

    Assumes:
    - Structured EHR data is encoded by `structured_encoder`.
    - Text notes are encoded by `text_encoder` (e.g., a pre-trained ClinicalBERT).
    - Both encoders produce fixed-size embeddings.
    """
    def __init__(self, structured_encoder, text_encoder,
                 structured_embedding_dim, text_embedding_dim,
                 projection_dim=128, temperature=0.07):
        super(MMSimCLR, self).__init__()

        self.structured_encoder = structured_encoder # This could be a simple MLP or more complex
        self.text_encoder = text_encoder # This could be a (potentially frozen) BERT model

        # Projection heads for both modalities
        self.structured_projector = ProjectionHead(structured_embedding_dim, output_dim=projection_dim)
        self.text_projector = ProjectionHead(text_embedding_dim, output_dim=projection_dim)

        self.temperature = temperature

    def forward(self, structured_data_batch, text_data_batch):
        """
        Performs a forward pass for contrastive learning.
        In SimCLR, typically two augmented views of the same sample are contrasted.
        In a multimodal setup like this (structured vs. text for the same patient visit),
        we contrast the projection of structured data with the projection of its paired text data.

        Args:
            structured_data_batch: A batch of structured EHR data.
                                   Shape: (batch_size, num_structured_features)
            text_data_batch: A batch of text note data (e.g., pre-computed embeddings or raw text
                             to be processed by self.text_encoder).
                             If pre-computed embeddings: Shape: (batch_size, text_embedding_dim)

        Returns:
            loss (torch.Tensor): The contrastive loss.
            z_struct (torch.Tensor): Projected embeddings for structured data.
            z_text (torch.Tensor): Projected embeddings for text data.
        """
        # 1. Encode each modality
        # Depending on text_encoder, text_data_batch might be raw text or tokenized ids
        # For simplicity, assume text_data_batch are already embeddings if text_encoder is an identity/projector
        # Or, text_encoder itself handles tokenization and embedding if it's a full BERT.

        # h_struct = self.structured_encoder(structured_data_batch) # (batch_size, structured_embedding_dim)
        # h_text = self.text_encoder(text_data_batch) # (batch_size, text_embedding_dim)

        # For this conceptual sketch, let's assume structured_data_batch and text_data_batch
        # are already the outputs of their respective encoders (h_struct, h_text).
        # In a full implementation, the encoders would be called here.
        h_struct = structured_data_batch # Assuming these are already encoded representations
        h_text = text_data_batch   # Assuming these are already encoded representations


        # 2. Project embeddings to the contrastive learning space
        z_struct = self.structured_projector(h_struct) # (batch_size, projection_dim)
        z_text = self.text_projector(h_text)     # (batch_size, projection_dim)

        # 3. Calculate contrastive loss (InfoNCE loss)
        # We want to maximize agreement between (z_struct_i, z_text_i) pairs
        # and minimize agreement with other pairs in the batch.

        loss = self.info_nce_loss(z_struct, z_text)

        return loss, z_struct, z_text

    def info_nce_loss(self, features_1, features_2):
        """
        Calculates InfoNCE loss between two sets of features (modalities).
        Assumes features_1[i] and features_2[i] are positive pairs.

        Args:
            features_1 (torch.Tensor): Embeddings from modality 1 (e.g., structured), normalized.
                                       Shape: (batch_size, projection_dim)
            features_2 (torch.Tensor): Embeddings from modality 2 (e.g., text), normalized.
                                       Shape: (batch_size, projection_dim)
        Returns:
            torch.Tensor: The InfoNCE loss.
        """
        batch_size = features_1.shape[0]
        device = features_1.device

        # Cosine similarity matrix (batch_size, batch_size)
        # sim_matrix_s1_s2[i, j] = similarity(features_1[i], features_2[j])
        sim_matrix_s1_s2 = torch.matmul(features_1, features_2.T) / self.temperature

        # Positive pairs are on the diagonal: sim(features_1[i], features_2[i])
        # We also need sim(features_2[i], features_1[i]), which is the same due to symmetry here.

        # For loss_s1_vs_s2: anchor=features_1, positive=features_2 (diagonal)
        # Denominator: sum_j exp(sim(features_1[i], features_2[j]))
        # Numerator: exp(sim(features_1[i], features_2[i]))

        # For loss_s2_vs_s1: anchor=features_2, positive=features_1 (diagonal)
        # Denominator: sum_j exp(sim(features_2[i], features_1[j]))
        # Numerator: exp(sim(features_2[i], features_1[i]))

        # Create labels: positive pairs are at index i for sample i.
        labels = torch.arange(batch_size, device=device) # [0, 1, 2, ..., batch_size-1]

        # Loss for structured modality using text as targets (and vice-versa)
        # loss_s1_s2 = F.cross_entropy(sim_matrix_s1_s2, labels)
        # loss_s2_s1 = F.cross_entropy(sim_matrix_s1_s2.T, labels) # Transpose for the other direction

        # A common way for two views (v1, v2) is to compute loss for (v1 -> v2) and (v2 -> v1)
        # sim_v1_v2 = (v1 @ v2.T) / self.temperature
        # sim_v2_v1 = (v2 @ v1.T) / self.temperature  # which is just sim_v1_v2.T

        # logits = sim_v1_v2
        # labels = torch.arange(batch_size).to(device)
        # loss_v1_v2 = F.cross_entropy(logits, labels)
        # loss_v2_v1 = F.cross_entropy(logits.T, labels) # or F.cross_entropy(sim_v2_v1, labels)
        # total_loss = (loss_v1_v2 + loss_v2_v1) / 2.0

        # Simplified: For each z_struct_i, z_text_i is positive, all other z_text_j are negative.
        # And for each z_text_i, z_struct_i is positive, all other z_struct_j are negative.

        # This implementation follows the typical NT-Xent loss formulation for two views
        # where (f1_i, f2_i) is a positive pair.
        # Concatenate all features to form negatives easily.
        # features = torch.cat([features_1, features_2], dim=0) # Shape: (2*batch_size, projection_dim)
        # sim_matrix = torch.matmul(features, features.T) / self.temperature # (2N, 2N)
        # # Mask out diagonal (self-similarity)
        # sim_matrix = sim_matrix - torch.eye(2 * batch_size, device=device) * 1e9

        # # Positive pairs are (i, i+N) and (i+N, i)
        # positives_row1 = torch.diag(sim_matrix, batch_size) # sim(f1_i, f2_i)
        # positives_row2 = torch.diag(sim_matrix, -batch_size) # sim(f2_i, f1_i)

        # This is simpler:
        loss_s1_s2 = F.cross_entropy(sim_matrix_s1_s2, labels)
        loss_s2_s1 = F.cross_entropy(sim_matrix_s1_s2.T, labels) # For text anchors, struct positives/negatives

        total_loss = (loss_s1_s2 + loss_s2_s1) / 2.0
        return total_loss


# --- Dummy Encoders for Demonstration ---
class DummyStructuredEncoder(nn.Module):
    def __init__(self, input_features, output_embedding_dim):
        super(DummyStructuredEncoder, self).__init__()
        self.fc = nn.Linear(input_features, output_embedding_dim)
    def forward(self, x):
        return F.relu(self.fc(x))

class DummyTextEncoder(nn.Module): # Assumes text_data is already pre-embedded
    def __init__(self, input_embedding_dim, output_embedding_dim):
        super(DummyTextEncoder, self).__init__()
        # If input_embedding_dim is different from output_embedding_dim, add a linear layer
        if input_embedding_dim != output_embedding_dim:
            self.fc = nn.Linear(input_embedding_dim, output_embedding_dim)
        else:
            self.fc = nn.Identity() # If pre-computed embeddings are already the desired dim

    def forward(self, x_embeddings):
        return self.fc(x_embeddings)


if __name__ == '__main__':
    print("--- MM-SimCLR Conceptual Example ---")

    # Configuration
    batch_size = 4 # Number of (structured, text) pairs
    num_structured_features = 50
    text_pre_embedding_dim = 768 # e.g., from ClinicalBERT

    # Encoder output dimensions (before projection head)
    structured_encoder_output_dim = 256
    text_encoder_output_dim = text_pre_embedding_dim # If DummyTextEncoder is Identity or simple projection

    projection_head_output_dim = 128 # Dimension of the space for contrastive loss
    temperature_param = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create dummy encoders
    # In a real scenario, structured_encoder could be an MLP, and
    # text_encoder could be a (possibly frozen) BERT model + pooling.
    # For this example, let's assume text_encoder takes pre-computed text embeddings.
    struct_enc = DummyStructuredEncoder(num_structured_features, structured_encoder_output_dim).to(device)

    # If text_encoder is meant to further process pre-computed embeddings:
    text_enc = DummyTextEncoder(text_pre_embedding_dim, text_encoder_output_dim).to(device)
    # If text_encoder were, e.g., a full BERT model, it would take tokenized input.

    # 2. Initialize MM-SimCLR model
    mm_simclr_model = MMSimCLR(
        structured_encoder=struct_enc, # This is not used if we pass encoded repr directly to forward
        text_encoder=text_enc,         # This is not used if we pass encoded repr directly to forward
        structured_embedding_dim=structured_encoder_output_dim, # Input dim for struct_projector
        text_embedding_dim=text_encoder_output_dim,             # Input dim for text_projector
        projection_dim=projection_head_output_dim,
        temperature=temperature_param
    ).to(device)

    # 3. Create dummy input data
    # These represent the *outputs* of the base encoders (h_struct, h_text)
    # In a full pipeline, you'd pass raw structured data and raw text (or tokens)
    # and the encoders within MMSimCLR would process them first.
    # For this example, we simulate having already encoded them.
    dummy_h_structured = torch.randn(batch_size, structured_encoder_output_dim, device=device)
    dummy_h_text = torch.randn(batch_size, text_encoder_output_dim, device=device) # Pre-computed text embeddings


    # 4. Forward pass
    # The current MMSimCLR.forward expects h_struct and h_text directly.
    # If it were to use its internal encoders, it would be:
    # loss, z_s, z_t = mm_simclr_model(raw_struct_data, raw_text_data)

    loss, z_s, z_t = mm_simclr_model(dummy_h_structured, dummy_h_text)

    print(f"\nMM-SimCLR Loss: {loss.item()}")
    print(f"Projected structured embeddings shape: {z_s.shape}") # (batch_size, projection_head_output_dim)
    print(f"Projected text embeddings shape: {z_t.shape}")       # (batch_size, projection_head_output_dim)

    # Conceptual: A training loop would involve:
    # - Loading batches of paired (structured_ehr, text_note) data.
    # - Encoding them using respective encoders (e.g., MLP for EHR, BERT for text).
    # - Passing these encoded representations (h_struct, h_text) to mm_simclr_model.forward().
    # - Calculating loss.
    # - Backpropagating and updating weights of encoders (if not frozen) and projection heads.

    # optimizer = torch.optim.Adam(mm_simclr_model.parameters(), lr=1e-3)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print("\nConceptual: Backward pass and optimizer step performed.")

    print("\n--- MM-SimCLR Example Finished ---")


