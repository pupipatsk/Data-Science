from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import Config


class DataProcessor:
    """A class for processing datasets."""

    def __init__(self, random_state: int = Config.SEED):
        """
        Initializes the DataProcessor class with default attributes.

        Args:
            random_state (int, optional): Seed for reproducibility. Defaults to 42.
        """
        self.df_dataset: Optional[pd.DataFrame] = None
        self.target: Optional[str] = None
        self.random_state: int = random_state

    def initial_train_test_split(
        self, df_dataset: pd.DataFrame, test_size: float = 0.10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and test sets.

        Args:
            df_dataset (pd.DataFrame): The input dataset.
            test_size (float, optional): Proportion of the dataset to be used as the test set. Defaults to 0.10.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets.
        """
        if not self.target:
            raise ValueError("Target column must be specified before splitting.")

        X = df_dataset.drop(columns=[self.target])
        y = df_dataset[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        # combine X and y
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return df_train, df_test

    def cut_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers based on quantile thresholds.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with outliers removed.
        """
        threshold = 50 / 500_000  # cut off 50(x2) samples from 500k to handle errors

        for col in df.select_dtypes(include=["number"]).columns:
            lower_bound = df[col].quantile(threshold)
            upper_bound = df[col].quantile(1 - threshold)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def normalize(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        num_cols: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalizes numerical features in the dataset using StandardScaler.

        Args:
            df_train (pd.DataFrame): Training dataset.
            df_test (pd.DataFrame): Test dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Normalized training and test datasets.
        """
        if num_cols is None:
            num_cols = df_train.select_dtypes(include=["number"]).columns.tolist()
        if self.target in num_cols:
            num_cols.remove(self.target)

        scaler = StandardScaler()
        df_train[num_cols] = scaler.fit_transform(df_train[num_cols].copy())
        df_test[num_cols] = scaler.transform(df_test[num_cols].copy())

        return df_train, df_test

    def process(
        self, df_dataset: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes the dataset by removing outliers, normalizing numerical features, and splitting into train/test sets.

        Args:
            df_dataset (pd.DataFrame): The dataset to be processed.
            target (str): The target column for prediction.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed train and test datasets.
        """
        if not target:
            raise ValueError("Target column must be specified.")

        self.df_dataset = df_dataset.copy()
        self.target = target

        # Drop ID
        if "id" in self.df_dataset.columns:
            self.df_dataset.drop(columns=["id"], inplace=True)

        # Cut off outliers
        df_dataset = self.cut_outliers(df_dataset)

        # Initial split: dataset → train (80+10%) | test (10%)
        df_train, df_test = self.initial_train_test_split(
            self.df_dataset, test_size=0.10
        )

        # Normalize
        df_train, df_test = self.normalize(df_train, df_test)

        return df_train, df_test
