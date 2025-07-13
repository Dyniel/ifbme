import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Recommended ClinicalBERT model: 'emilyalsentzer/Bio_ClinicalBERT' or 'medicalai/ClinicalBERT'
# or specific MIMIC fine-tuned ones like 'biobert-base-mimic'.
# Using a general one for broader applicability if specific MIMIC model isn't available/required by user.
DEFAULT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'

def get_clinical_bert_model_and_tokenizer(model_name=DEFAULT_MODEL_NAME, device=None):
    """
    Loads a pre-trained ClinicalBERT model and tokenizer.

    Args:
        model_name (str): The name of the ClinicalBERT model from Hugging Face Model Hub.
        device (torch.device, optional): Device to load the model onto ('cuda' or 'cpu').
                                         If None, uses CUDA if available, else CPU.

    Returns:
        tuple: (tokenizer, model)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model: {model_name} to device: {device}")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval() # Set to evaluation mode by default
    return tokenizer, model

def fine_tune_clinical_bert(model, tokenizer, train_texts, train_labels, epochs=2, lr=2e-5, device=None):
    """
    Placeholder for fine-tuning ClinicalBERT on specific clinical notes.
    Actual fine-tuning requires a proper dataset, training loop, optimizer, and loss function.

    Args:
        model: The ClinicalBERT model to fine-tune.
        tokenizer: The tokenizer for the model.
        train_texts (list of str): List of clinical notes for training.
        train_labels (list or np.array): Corresponding labels for a downstream task (e.g., mortality).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (torch.device): Device to perform training on.

    Returns:
        model: The fine-tuned model.
    """
    print(f"Conceptual: Starting fine-tuning of ClinicalBERT for {epochs} epochs with lr={lr} on {device}.")
    print(f"Received {len(train_texts)} texts for fine-tuning.")

    # --- This is a highly simplified placeholder ---
    # A real fine-tuning loop would involve:
    # 1. Preparing a PyTorch Dataset and DataLoader for train_texts and train_labels.
    #    - Tokenizing texts, creating attention masks, handling padding/truncation.
    #    - Formatting labels appropriately for the task (e.g., classification head).
    # 2. Adding a task-specific head to the BERT model if it's for classification/regression.
    #    (e.g., model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N))
    # 3. Defining a loss function (e.g., CrossEntropyLoss).
    # 4. Defining an optimizer (e.g., AdamW).
    # 5. Iterating through epochs and batches:
    #    - model.train()
    #    - Forward pass: outputs = model(input_ids, attention_mask, labels=batch_labels)
    #    - Calculate loss: loss = outputs.loss
    #    - Backward pass: loss.backward()
    #    - Optimizer step: optimizer.step()
    #    - Zero gradients: optimizer.zero_grad()
    # 6. Optional: Validation loop, learning rate scheduling.

    print("Conceptual: Fine-tuning complete. Returning model (in this placeholder, it's the original model).")
    return model


def get_text_embeddings(texts, tokenizer, model, strategy='cls', max_length=512, batch_size=32, device=None):
    """
    Generates embeddings for a list of texts using the provided ClinicalBERT model.

    Args:
        texts (list of str): A list of clinical notes or text snippets.
        tokenizer: The pre-trained tokenizer.
        model: The pre-trained (or fine-tuned) ClinicalBERT model.
        strategy (str): The pooling strategy for obtaining a fixed-size embedding from token embeddings.
                        'cls': Use the embedding of the [CLS] token.
                        'mean': Use the mean of all token embeddings in the last hidden state.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Number of texts to process in one batch.
        device (torch.device, optional): Device to perform inference on.
                                         If None, uses model's device.

    Returns:
        np.ndarray: A 2D numpy array of shape (num_texts, embedding_dim) containing the embeddings.
                    Returns None if input texts is empty or None.
    """
    if not texts:
        return None

    if device is None:
        device = model.device

    model.eval() # Ensure model is in evaluation mode
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_dim)

        if strategy == 'cls':
            # [CLS] token is at the beginning of the sequence
            batch_embeddings = last_hidden_states[:, 0, :].cpu().numpy()
        elif strategy == 'mean':
            # Mean pooling, careful about padding tokens if attention_mask is not used by model directly for output
            # For mean pooling, we should only consider non-padding tokens.
            # The attention_mask from the tokenizer can be used here.
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        else:
            raise ValueError(f"Unsupported pooling strategy: {strategy}. Choose 'cls' or 'mean'.")

        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


if __name__ == '__main__':
    # --- Example Usage ---
    print("--- ClinicalBERT Embedding Example ---")

    # 1. Load model and tokenizer
    # Uses default model 'emilyalsentzer/Bio_ClinicalBERT'
    # This will download the model if not cached, may take time.
    try:
        tokenizer, model = get_clinical_bert_model_and_tokenizer()
        bert_device = model.device
        print(f"Model and tokenizer loaded successfully onto {bert_device}.")
    except Exception as e:
        print(f"Error loading default ClinicalBERT model: {e}")
        print("Skipping embedding generation example. Ensure you have internet access "
              "and the Hugging Face model name is correct.")
        exit()

    # 2. Sample clinical notes
    sample_notes = [
        "Patient presents with fever and cough. Chest X-ray shows bilateral infiltrates.",
        "History of hypertension and type 2 diabetes. Current medications include metformin and lisinopril.",
        "No acute distress. Vital signs stable. Plan to discharge tomorrow.",
        "", # Empty string
        "Short note."
    ]
    print(f"\nSample notes (first 2): {sample_notes[:2]}")

    # 3. Generate embeddings (using CLS token strategy)
    print("\nGenerating embeddings with 'cls' strategy...")
    embeddings_cls = get_text_embeddings(sample_notes, tokenizer, model, strategy='cls', device=bert_device)
    if embeddings_cls is not None:
        print(f"CLS Embeddings shape: {embeddings_cls.shape}") # Should be (num_notes, 768) for base BERT models
        # print("CLS Embedding for first note (first 10 dims):", embeddings_cls[0, :10])

    # 4. Generate embeddings (using mean pooling strategy)
    print("\nGenerating embeddings with 'mean' strategy...")
    embeddings_mean = get_text_embeddings(sample_notes, tokenizer, model, strategy='mean', device=bert_device)
    if embeddings_mean is not None:
        print(f"Mean Pool Embeddings shape: {embeddings_mean.shape}")
        # print("Mean Pool Embedding for first note (first 10 dims):", embeddings_mean[0, :10])

    # 5. Conceptual Fine-tuning (placeholder)
    # To actually run fine-tuning, you'd need a labeled dataset.
    # For this example, we'll just call the placeholder.
    print("\nConceptual fine-tuning call:")
    dummy_labels = [0] * len(sample_notes) # Dummy labels for the placeholder
    # model = fine_tune_clinical_bert(model, tokenizer, sample_notes, dummy_labels, epochs=1, device=bert_device)
    print("Fine-tuning placeholder executed.")

    # 6. Test with empty list of texts
    print("\nTesting with empty list of texts:")
    empty_embeddings = get_text_embeddings([], tokenizer, model)
    if empty_embeddings is None or empty_embeddings.shape[0] == 0 :
        print("Correctly handled empty list, returned None or empty array.")
    else:
        print(f"Error: Expected None or empty array for empty list, got shape {empty_embeddings.shape}")

    print("\n--- Example Finished ---")