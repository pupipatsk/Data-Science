from typing import Optional, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.config import Config
from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    A Logistic Regression model wrapper extending BaseModel.

    Attributes:
        model (LogisticRegression): The logistic regression model instance.
        base_params (Dict[str, Any]): Default model parameters.
        learnable_params (Dict[str, Dict[str, Any]]): Parameter space for hyperparameter tuning.
        params (Dict[str, Any]): Final set of parameters used for training.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initializes the LogisticRegressionModel class with default parameters.

        Args:
            random_state (Optional[int], optional): Random seed for model reproducibility.
                Defaults to `Config.SEED` if not provided.
        """
        super().__init__()

        # Ensure Config.SEED is available before using it
        if random_state is None:
            random_state = getattr(
                Config, "SEED", 42
            )  # Default to 42 if Config.SEED is not found

        self.model = LogisticRegression(random_state=random_state)
        self.base_params: Dict[str, Any] = {"random_state": random_state}
        self.learnable_params: Dict[str, Dict[str, Any]] = {
            "C": {
                "type": "loguniform",
                "low": 0.01,
                "high": 10,
            }  # Regularization strength
        }
        self.params: Dict[str, Any] = self.base_params.copy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the logistic regression model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Raises:
            ValueError: If `X` or `y` are not provided correctly.
        """
        if X is None or y is None:
            raise ValueError("Training data (X, y) cannot be None.")

        self.model.set_params(**self.params)
        self.model.fit(X, y)
