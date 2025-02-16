from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseModel:
    """
    A base class for machine learning models that provides fit, predict, and evaluation functionalities.

    Attributes:
        model (Optional[Any]): The machine learning model instance.
        params (Dict[str, Any]): Model hyperparameters.
        metrics (Dict[str, callable]): Dictionary of evaluation metrics.
    """

    def __init__(self):
        """
        Initializes the BaseModel class with default attributes.
        """
        self.model: Optional[Any] = None
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, callable] = {
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "accuracy": accuracy_score,
            "roc_auc": roc_auc_score,
        }  # {metric_name: metric_function}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Please assign a valid model to `self.model`."
            )

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Please assign a valid model to `self.model`."
            )

        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
        """
        Evaluates the model on a given dataset using predefined metrics.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True target labels of shape (n_samples,).

        Returns:
            Dict[str, Optional[float]]: A dictionary containing evaluation metric names as keys and their computed values as values.
                                        Returns `None` for metrics that could not be computed.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Please assign a valid model to `self.model`."
            )

        y_pred = self.predict(X)
        results: Dict[str, Optional[float]] = {}

        for metric_name, metric_fn in self.metrics.items():
            try:
                if metric_name == "roc_auc":
                    if hasattr(self.model, "predict_proba"):
                        y_prob = self.model.predict_proba(X)[
                            :, 1
                        ]  # Get probabilities for the positive class (1)
                        results[metric_name] = metric_fn(y, y_prob)
                    else:
                        results[metric_name] = (
                            None  # Model does not support probability estimates
                        )
                else:
                    results[metric_name] = metric_fn(y, y_pred)
            except ValueError as e:
                print(f"Warning: Could not compute {metric_name} due to error: {e}")
                results[metric_name] = None

        return results
