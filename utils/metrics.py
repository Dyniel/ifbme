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
        pass # Allow sum to propagate NaN if individual scores are NaN despite handling
    return dt_score + ls_score
