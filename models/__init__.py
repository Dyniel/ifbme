# This directory will store the implementations of our predictive models:
# - STM-GNN (Space-Time-Memory Graph Neural Network)
# - LightGBM
# - TECO-Transformer
# - XGBoost (Meta-learner)

# Expose key models or layers for easier import
from .stm_gnn import STMGNN, STMGNNLayer
# from .lightgbm_model import LightGBMModel # Example for future
# from .teco_transformer import TECOTransformerModel # Example for future
# from .meta_learner import XGBoostMetaLearner # Example for future

# Potentially, the model that integrates STMGNNLayer as per plan:
from .main_model import ModelWithSTMGNNLayer

__all__ = [
    "STMGNN",
    "STMGNNLayer",
    "ModelWithSTMGNNLayer",
    # "LightGBMModel",
    # "TECOTransformerModel",
    # "XGBoostMetaLearner"
]
