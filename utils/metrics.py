import numpy as np


def dt_score_calc(f1_death_class_score: float) -> float:
    """
    Calculates the Discharge Type Score (DTscore).
    DTscore = 10 * (1 - F1-score_death_class)
    A lower DTscore is better. Perfect score is 0, worst is 10.
    """
    if np.isnan(f1_death_class_score):
        # If F1 score is NaN (e.g., no true positives and no predicted positives for death class),
        # it indicates a failure for that class. Default to worst score.
        return 10.0
    return 10 * (1 - f1_death_class_score)


def ls_score_calc(mae_los_score: float) -> float:
    """
    Calculates the Length of Stay Score (LSscore).
    LSscore = min(10, MAE_LS)
    A lower LSscore is better. Perfect score is 0, capped at 10.
    """
    if np.isnan(mae_los_score):
        # If MAE is NaN, it indicates a failure in LoS prediction for the fold.
        return 10.0
    return min(10.0, mae_los_score)


def gl_score_calc(dt_score: float, ls_score: float) -> float:
    """
    Calculates the Global Score (GLscore).
    GLscore = DTscore + LSscore
    A lower GLscore is better.
    """
    # DTscore and LSscore should already be handled for NaNs by their respective functions.
    # However, if they still could be NaN, this ensures GLscore also reflects that.
    if np.isnan(dt_score) or np.isnan(ls_score):
        # This case should ideally not be reached if dt_score_calc and ls_score_calc handle NaNs.
        # Returning NaN or a very high number could be options.
        # Let's assume components are non-NaN; if not, sum will propagate NaN.
        # If one is 10 (worst) and other is valid, sum is still meaningful.
        pass  # Allow sum to propagate NaN if individual scores are NaN despite handling
    return dt_score + ls_score


def maximise_f1_threshold(y_true: np.ndarray, y_probas: np.ndarray, target_label_value: int, class_mapping: dict,
                          positive_label_name: str = 'Death'):
    """
    Finds the optimal probability threshold for a specific class to maximize its F1 score.

    Args:
        y_true (np.ndarray): True binary labels.
        y_probas (np.ndarray): Predicted probabilities for all classes (shape [n_samples, n_classes]).
        target_label_value (int): The integer value in y_true that represents the positive class
                                   for which F1 is being maximized (e.g., the encoded value for 'Death').
        class_mapping (dict): A dictionary mapping class names to their encoded integer values
                              (e.g., {'Death': 0, 'Survival': 1}).
        positive_label_name (str): The name of the positive class (e.g., "Death").

    Returns:
        tuple: (best_threshold, max_f1_score)
               Returns (0.5, 0.0) if the positive class is not found in class_mapping or other issues occur.
    """
    from sklearn.metrics import f1_score

    if positive_label_name not in class_mapping:
        # logger.warning(f"Positive label '{positive_label_name}' not found in class_mapping. Cannot maximize F1.")
        # Not ideal to have logger here if this is a pure util. Calling code should log.
        return 0.5, 0.0  # Default threshold and worst F1

    positive_class_idx = -1
    # Find the column index for the probabilities of the positive class
    # This depends on how y_probas columns are ordered relative to class_mapping values
    # Assuming y_probas columns directly correspond to sorted unique label values (0, 1, ..., n_classes-1)
    # and class_mapping gives the encoded value for positive_label_name.
    # Example: class_mapping = {'Death': 0, 'Survival': 1}. If positive_label_name is 'Death',
    # its encoded value is 0. So we need probabilities from column 0 of y_probas.

    # Find the index in y_probas corresponding to the positive_label_name
    # This is tricky if y_probas column order isn't guaranteed to map directly to sorted class_mapping values.
    # For safety, let's assume y_probas columns are ordered 0, 1, ... num_classes-1,
    # and target_label_value IS the index for y_probas.
    # If class_mapping = {'Death': 0, 'Survival': 1}, and target_label_value for Death is 0, then y_probas[:, 0] is P(Death).
    # If class_mapping = {'Survival': 0, 'Death': 1}, and target_label_value for Death is 1, then y_probas[:, 1] is P(Death).

    # The `target_label_value` itself should be the index if y_probas is ordered by class indices.
    # However, the prompt implies target_label_value is what's in y_true.
    # Let's clarify: class_mapping maps names to numerical labels used in y_true.
    # y_probas has shape (n_samples, n_classes). Typically, column k is P(class_k).
    # So, if 'Death' is encoded as 0 in y_true (target_label_value=0), then y_probas[:, 0] is P(Death).

    prob_col_idx = class_mapping.get(positive_label_name)
    if prob_col_idx is None or prob_col_idx >= y_probas.shape[1]:
        # Fallback or error if class name not in mapping or index out of bounds
        # This check is slightly redundant due to the initial check, but good for safety.
        return 0.5, 0.0

    y_probas_positive_class = y_probas[:, prob_col_idx]

    best_threshold = 0.5
    max_f1 = 0.0

    # Define thresholds to check
    thresholds = np.linspace(0.01, 0.99, 100)

    for threshold in thresholds:
        # Predict based on current threshold for the positive class
        # If proba_positive_class > threshold, predict positive_label_value, else predict the other label.
        # For F1 score, we need y_pred to be in terms of the original labels (e.g., 0s and 1s from y_true)
        # where target_label_value is the positive class.

        # Convert y_true to binary 0/1 format where 1 is the positive class (target_label_value)
        y_true_binary = (y_true == target_label_value).astype(int)

        # Predictions are 1 if proba > threshold, else 0. This aligns with y_true_binary.
        y_pred_binary_at_threshold = (y_probas_positive_class > threshold).astype(int)

        # Calculate F1 score. We are interested in F1 for target_label_value.
        # Since y_true_binary and y_pred_binary_at_threshold are already 0/1 for our target class,
        # pos_label=1 is appropriate here.
        current_f1 = f1_score(y_true_binary, y_pred_binary_at_threshold, pos_label=1, zero_division=0)

        if current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = threshold

    return best_threshold, max_f1
