from typing import Dict, Optional
import os
import time
import joblib
import pandas as pd
from .base_model import BaseModel
from .single_trainer import SingleTrainer


class MultiTrainer:
    """
    A class for training and evaluating multiple models on a given dataset.

    Attributes:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Test dataset.
        target (str): Target variable name.
        models (Dict[str, BaseModel]): Dictionary of models to train.
        main_metric (str): Metric used for model evaluation.
        verbose (bool): Whether to print progress messages.
        output_dir (Optional[str]): Directory to save trained models.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target: str,
        models: Dict[str, BaseModel],
        main_metric: str,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the MultiTrainer class.

        Args:
            df_train (pd.DataFrame): Training dataset.
            df_test (pd.DataFrame): Test dataset.
            target (str): Target variable for prediction.
            models (Dict[str, BaseModel]): Dictionary of models to train.
            main_metric (str): The primary metric used for model evaluation.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            output_dir (Optional[str], optional): Directory to save trained models. Defaults to None.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.target = target
        self.models: Dict[str, BaseModel] = models  # {model_name: model_instance}
        self.trained_models: Dict[str, BaseModel] = {}
        self.main_metric = main_metric
        self.verbose = verbose
        self.output_dir = output_dir

    @staticmethod
    def _save_model(
        model: BaseModel,
        output_dir: str,
        file_format: str = "pkl",
        verbose: bool = True,
    ) -> None:
        """
        Saves a trained model to a specified directory.

        Args:
            model (BaseModel): The trained model instance.
            output_dir (str): Directory where the model will be saved.
            file_format (str, optional): Format for saving the model. Defaults to "pkl".
            verbose (bool, optional): Whether to print save location. Defaults to True.
        """
        if output_dir is None:
            raise ValueError("output_dir cannot be None when saving models.")

        time_now = time.strftime("%Y-%m-%d-%H%M")
        model_name = model.__class__.__name__
        file_name = f"{time_now}-{model_name}"

        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(output_dir, f"{file_name}.{file_format}")

        joblib.dump(model, file_path)
        if verbose:
            print(f"Model saved to {file_path}")

    def train_all_models(self, tune_params: bool = False) -> None:
        """
        Trains all models and optionally tunes hyperparameters.

        Args:
            tune_params (bool, optional): If True, tunes model hyperparameters. Defaults to False.
        """
        for model_name, model in self.models.items():
            if self.verbose:
                print(f"Training {model_name}...")
                start_time = time.time()

            single_trainer = SingleTrainer(model, self.main_metric)
            single_trainer.train(self.df_train, self.target, tune_params)
            self.trained_models[model_name] = single_trainer.model

            # Save model if output_dir is specified
            if self.output_dir:
                self._save_model(
                    single_trainer.model, self.output_dir, verbose=self.verbose
                )

            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f"Training time {model_name}: {elapsed_time:.2f} seconds.")

    def evaluate_all_models(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluates all trained models on both train and test datasets.

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]:
                A dictionary containing evaluation results in the format:
                {
                    "train": {model_name: {metric_name: value, ...}, ...},
                    "test": {model_name: {metric_name: value, ...}, ...}
                }
        """
        results: Dict[str, Dict[str, Dict[str, float]]] = {"train": {}, "test": {}}

        for dataset_name, df in [("train", self.df_train), ("test", self.df_test)]:
            X = df.drop(columns=[self.target])
            y = df[self.target]

            for model_name, model in self.trained_models.items():
                res = model.evaluate(X, y)
                score = res.get(self.main_metric)  # Avoids KeyError

                results[dataset_name][model_name] = res
                if self.verbose:
                    score_display = (
                        f"{score:.4f}" if score is not None else "Metric not available"
                    )
                    print(
                        f"{dataset_name.upper()} | {model_name} {self.main_metric}: {score_display}"
                    )

        return results
