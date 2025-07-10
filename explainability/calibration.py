import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split # For example usage

class ModelCalibrator:
    """
    Calibrates model probability outputs using Isotonic Regression.
    Isotonic Regression is a non-parametric method that fits a non-decreasing
    function to the probability scores.
    """
    def __init__(self):
        self.calibrators = {} # To store one calibrator per class for multiclass

    def fit(self, y_true, y_pred_proba, n_classes=None):
        """
        Fits the Isotonic Regression calibrator(s).

        Args:
            y_true (np.ndarray): True labels (integers 0 to K-1). Shape (n_samples,).
            y_pred_proba (np.ndarray): Predicted probabilities from the uncalibrated model.
                                       Shape (n_samples, n_classes) for multiclass,
                                       or (n_samples,) for binary positive class probability.
            n_classes (int, optional): Number of classes. If None, inferred from y_pred_proba.
                                       Required if y_pred_proba is 1D for binary case but needs
                                       to be treated as 2-class internally.
        """
        if y_pred_proba.ndim == 1: # Binary case, probabilities of the positive class
            if n_classes is None or n_classes <= 1:
                n_classes = 2 # Assume 2 classes if 1D proba

            # Convert to (n_samples, 2) format for consistency if needed,
            # but IsotonicRegression can take 1D y_pred for binary.
            # For this implementation, we'll calibrate the positive class probability directly.
            self.calibrators[1] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            # For binary, y_true should be 0 or 1.
            self.calibrators[1].fit(y_pred_proba, y_true)
            # Store n_classes for predict_proba logic
            self._n_classes_at_fit = 2

        else: # Multiclass case
            self._n_classes_at_fit = y_pred_proba.shape[1]
            if n_classes is not None and n_classes != self._n_classes_at_fit:
                raise ValueError("Provided n_classes does not match y_pred_proba columns.")

            for k in range(self._n_classes_at_fit):
                # For each class k, we calibrate its predicted probability P(y=k|X).
                # The "true" value for P(y=k|X) is 1 if y_true == k, and 0 otherwise.
                y_true_k = (y_true == k).astype(int)

                calibrator_k = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                calibrator_k.fit(y_pred_proba[:, k], y_true_k)
                self.calibrators[k] = calibrator_k

        print(f"Isotonic calibrators fitted for {self._n_classes_at_fit} class(es).")

    def predict_proba(self, y_pred_proba_uncalibrated):
        """
        Applies the fitted Isotonic Regression to get calibrated probabilities.

        Args:
            y_pred_proba_uncalibrated (np.ndarray): Uncalibrated predicted probabilities.
                                                   Shape (n_samples, n_classes) or (n_samples,).

        Returns:
            np.ndarray: Calibrated probabilities. Shape (n_samples, n_classes).
        """
        if not self.calibrators:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        if y_pred_proba_uncalibrated.ndim == 1: # Binary case (positive class proba)
            if self._n_classes_at_fit != 2:
                raise ValueError("Calibrator was fitted for multiclass, but 1D probabilities provided.")

            calibrated_proba_pos_class = self.calibrators[1].predict(y_pred_proba_uncalibrated)
            # Reconstruct (n_samples, 2) shape
            calibrated_probas = np.zeros((len(y_pred_proba_uncalibrated), 2))
            calibrated_probas[:, 1] = calibrated_proba_pos_class
            calibrated_probas[:, 0] = 1 - calibrated_proba_pos_class
            return calibrated_probas

        else: # Multiclass case
            if y_pred_proba_uncalibrated.shape[1] != self._n_classes_at_fit:
                raise ValueError("Uncalibrated probabilities have different number of classes than fitter.")

            calibrated_probas = np.zeros_like(y_pred_proba_uncalibrated)
            for k in range(self._n_classes_at_fit):
                calibrated_probas[:, k] = self.calibrators[k].predict(y_pred_proba_uncalibrated[:, k])

            # Normalize probabilities to sum to 1 across classes (Isotonic Regressor per class doesn't guarantee this)
            sum_probas = np.sum(calibrated_probas, axis=1, keepdims=True)
            # Avoid division by zero if all calibrated probabilities for a sample are zero
            sum_probas[sum_probas == 0] = 1.0
            calibrated_probas_normalized = calibrated_probas / sum_probas

            return calibrated_probas_normalized

    def predict(self, y_pred_proba_uncalibrated):
        """Predicts class labels based on calibrated probabilities."""
        calibrated_probas = self.predict_proba(y_pred_proba_uncalibrated)
        return np.argmax(calibrated_probas, axis=1)


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression # Example uncalibrated model
    from sklearn.metrics import brier_score_loss, log_loss
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    print("--- ModelCalibrator (Isotonic Regression) Example ---")

    # 1. Generate dummy binary classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_classes=2, random_state=42, weights=[0.8, 0.2]) # Imbalanced

    # Split into train, validation (for calibration), and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 2. Train a simple model (e.g., Logistic Regression) - often miscalibrated
    print("\nTraining a Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Get uncalibrated probabilities on the calibration set
    y_pred_proba_calib_uncal = model.predict_proba(X_calib) # (n_samples, 2)
    y_pred_proba_test_uncal = model.predict_proba(X_test)

    # 3. Fit the Isotonic Calibrator
    print("\nFitting Isotonic Calibrator...")
    calibrator = ModelCalibrator()
    # For binary, can pass positive class probas (y_pred_proba_calib_uncal[:, 1])
    # or full probas. Let's test with full probas first.
    calibrator.fit(y_calib, y_pred_proba_calib_uncal)
    # calibrator.fit(y_calib, y_pred_proba_calib_uncal[:,1]) # Test with 1D probas for binary

    # 4. Get calibrated probabilities on the test set
    print("\nApplying calibration to test set probabilities...")
    y_pred_proba_test_calibrated = calibrator.predict_proba(y_pred_proba_test_uncal)
    # y_pred_proba_test_calibrated_from_1d = calibrator.predict_proba(y_pred_proba_test_uncal[:,1])


    # 5. Evaluate calibration (e.g., Brier score, reliability diagram)
    # Using positive class probabilities for binary metrics
    brier_uncal = brier_score_loss(y_test, y_pred_proba_test_uncal[:, 1])
    brier_cal = brier_score_loss(y_test, y_pred_proba_test_calibrated[:, 1])
    print(f"Brier Score (Uncalibrated): {brier_uncal:.4f}")
    print(f"Brier Score (Calibrated):   {brier_cal:.4f} (Lower is better)")

    logloss_uncal = log_loss(y_test, y_pred_proba_test_uncal)
    logloss_cal = log_loss(y_test, y_pred_proba_test_calibrated)
    print(f"Log Loss (Uncalibrated): {logloss_uncal:.4f}")
    print(f"Log Loss (Calibrated):   {logloss_cal:.4f} (Lower is better)")


    # Reliability Diagram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_pred_proba_test_uncal[:, 1], n_bins=10, strategy='uniform')
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', linewidth=1, label='Uncalibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel("Mean Predicted Probability (Uncalibrated)")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Uncalibrated)")
    plt.legend()

    plt.subplot(1, 2, 2)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_pred_proba_test_calibrated[:, 1], n_bins=10, strategy='uniform')
    plt.plot(prob_pred_cal, prob_true_cal, marker='o', linewidth=1, label='Calibrated (Isotonic)')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel("Mean Predicted Probability (Calibrated)")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Calibrated)")
    plt.legend()

    plt.tight_layout()
    # plt.show() # Comment out for non-interactive runs
    print("Reliability diagrams plotted (if GUI available).")


    # --- Multiclass Example ---
    print("\n--- Multiclass Calibration Example ---")
    X_mc, y_mc = make_classification(n_samples=1500, n_features=25, n_informative=15,
                                     n_classes=4, random_state=43, weights=[0.4,0.3,0.2,0.1])
    X_mc_train, X_mc_temp, y_mc_train, y_mc_temp = train_test_split(X_mc, y_mc, test_size=0.6, random_state=43, stratify=y_mc)
    X_mc_calib, X_mc_test, y_mc_calib, y_mc_test = train_test_split(X_mc_temp, y_mc_temp, test_size=0.5, random_state=43, stratify=y_mc_temp)

    model_mc = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=43)
    model_mc.fit(X_mc_train, y_mc_train)

    y_pred_proba_mc_calib_uncal = model_mc.predict_proba(X_mc_calib)
    y_pred_proba_mc_test_uncal = model_mc.predict_proba(X_mc_test)

    calibrator_mc = ModelCalibrator()
    calibrator_mc.fit(y_mc_calib, y_pred_proba_mc_calib_uncal)
    y_pred_proba_mc_test_calibrated = calibrator_mc.predict_proba(y_pred_proba_mc_test_uncal)

    # Check if calibrated probabilities sum to 1 for multiclass
    # print("Sum of calibrated probabilities for first 5 multiclass samples:", np.sum(y_pred_proba_mc_test_calibrated[:5], axis=1))
    assert np.allclose(np.sum(y_pred_proba_mc_test_calibrated, axis=1), 1.0), "Calibrated multiclass probabilities do not sum to 1."

    logloss_mc_uncal = log_loss(y_mc_test, y_pred_proba_mc_test_uncal)
    logloss_mc_cal = log_loss(y_mc_test, y_pred_proba_mc_test_calibrated)
    print(f"Multiclass Log Loss (Uncalibrated): {logloss_mc_uncal:.4f}")
    print(f"Multiclass Log Loss (Calibrated):   {logloss_mc_cal:.4f}")
    print("Note: ECE (Expected Calibration Error) is another common metric for multiclass calibration.")

    print("\n--- ModelCalibrator Example Finished ---")

```
