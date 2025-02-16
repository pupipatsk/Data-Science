from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.base_model import BaseModel
from src.config import Config


class XGBoostModel(BaseModel):
    """
    A wrapper for XGBoost's XGBClassifier extending BaseModel.

    Attributes:
        model (xgb.XGBClassifier): The XGBoost classifier model.
        base_params (Dict[str, Any]): Default model parameters.
        learnable_params (Dict[str, Dict[str, Any]]): Hyperparameter search space for tuning.
        params (Dict[str, Any]): Final set of parameters used for training.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initializes the XGBoostModel with default parameters.

        Args:
            random_state (Optional[int], optional): Random seed for reproducibility.
                Defaults to `Config.SEED` if not provided.
        """
        super().__init__()

        # Ensure Config.SEED is available before using it
        if random_state is None:
            random_state = getattr(
                Config, "SEED", 42
            )  # Default to 42 if Config.SEED is missing

        self.model = xgb.XGBClassifier(random_state=random_state)
        self.base_params: Dict[str, Any] = {
            "random_state": random_state,
            "objective": "binary:logistic",
        }
        self.learnable_params: Dict[str, Dict[str, Any]] = {
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "n_estimators": {"type": "int", "low": 50, "high": 300},
        }
        self.params: Dict[str, Any] = self.base_params.copy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the XGBoost model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Raises:
            ValueError: If `X` or `y` is None or empty.
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("Training data (X, y) cannot be None or empty.")

        self.model.set_params(**self.params)
        self.model.fit(np.array(X), np.array(y))
