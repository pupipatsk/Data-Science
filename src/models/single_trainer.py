from typing import Dict, Any, List
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from src.models.base_model import BaseModel
from src.config import Config


class SingleTrainer:
    """
    A class for training and hyperparameter tuning of a single machine learning model.

    Attributes:
        model (BaseModel): An instance of a machine learning model.
        target (str): The target variable name.
        main_metric (str): The main evaluation metric for model performance.
    """

    def __init__(self, model: BaseModel, main_metric: str):
        """
        Initializes the SingleTrainer class.

        Args:
            model (BaseModel): The machine learning model to be trained.
            main_metric (str): The primary metric to optimize during training.
        """
        self.model = model  # Custom model instance (e.g., LogisticRegressionModel)
        self.target: str = ""
        self.main_metric: str = main_metric

    def train(
        self, df_train: pd.DataFrame, target: str, tune_params: bool = False
    ) -> None:
        """
        Trains the model using the given dataset.

        Args:
            df_train (pd.DataFrame): The training dataset.
            target (str): The name of the target variable.
            tune_params (bool, optional): If True, performs hyperparameter tuning before training. Defaults to False.

        Raises:
            ValueError: If the target column is missing in df_train.
        """
        if target not in df_train.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset.")

        self.target = target
        X = df_train.drop(columns=[self.target]).values
        y = df_train[self.target].values

        if tune_params:
            best_params = self.tune_hyperparameters(df_train)
            self.model.model.set_params(**best_params)

        self.model.fit(X, y)

    def _retrieve_search_space(
        self, model: BaseModel, trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Retrieves and suggests hyperparameter search space for Optuna tuning.

        Args:
            model (BaseModel): The machine learning model.
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            Dict[str, Any]: Suggested hyperparameters.
        """
        trial_params = {}
        for param, search_space in model.learnable_params.items():
            if search_space["type"] == "int":
                trial_params[param] = trial.suggest_int(
                    param, search_space["low"], search_space["high"]
                )
            elif search_space["type"] == "loguniform":
                trial_params[param] = trial.suggest_loguniform(
                    param, search_space["low"], search_space["high"]
                )
        return trial_params

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]
    ) -> List[float]:
        """
        Performs cross-validation and returns fold scores.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            params (Dict[str, Any]): Hyperparameters to be used for training.

        Returns:
            List[float]: A list of scores for each fold.
        """
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        fold_scores = []

        for train_index, val_index in cv.split(X, y):
            x_tr, x_val = X[train_index], X[val_index]
            y_tr, y_val = y[train_index], y[val_index]

            model = type(self.model)()  # Re-instantiate model
            if hasattr(model, "model") and model.model is not None:
                model.model.set_params(**params)
                model.fit(x_tr, y_tr)
                score = model.evaluate(x_val, y_val).get(self.main_metric, None)
                if score is not None:
                    fold_scores.append(score)

        return fold_scores

    def objective(self, trial: optuna.Trial, df_train: pd.DataFrame) -> float:
        """
        Optuna objective function for hyperparameter tuning.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            df_train (pd.DataFrame): The dataset used for tuning.

        Returns:
            float: The mean cross-validation score.
        """
        trial_params = self._retrieve_search_space(self.model, trial)
        X = df_train.drop(columns=[self.target]).values
        y = df_train[self.target].values
        fold_scores = self._cross_validate(X, y, trial_params)
        return np.mean(fold_scores) if fold_scores else float("-inf")

    def tune_hyperparameters(
        self, df_train: pd.DataFrame, n_trials: int = 3
    ) -> Dict[str, Any]:
        """
        Optimizes hyperparameters using Optuna.

        Args:
            df_train (pd.DataFrame): The dataset used for tuning.
            n_trials (int, optional): Number of trials for hyperparameter tuning. Defaults to 3.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, df_train), n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters found: {best_params}")  # Logging the best params
        return best_params
