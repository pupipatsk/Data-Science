import os
import time
import numpy as np
import pandas as pd


def optimize_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimizes the memory usage of a pandas DataFrame by converting columns to the smallest possible data types.

    Args:
        df (pd.DataFrame): The input DataFrame to be optimized.
        verbose (bool, optional): Whether to print memory reduction details. Defaults to True.

    Returns:
        pd.DataFrame: The optimized DataFrame with reduced memory usage.
    """
    initial_mem = df.memory_usage().sum() / 1024**2  # Convert bytes to MB

    for col in df.columns:
        col_type = df[col].dtype

        # Check for NaN values before computing min and max
        if df[col].isnull().all():
            continue

        c_min, c_max = df[col].min(), df[col].max()

        # Convert numeric columns to optimal data types
        if np.issubdtype(col_type, np.integer):
            if c_min >= 0:
                if c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

        elif np.issubdtype(col_type, np.floating):
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

        # Convert object columns to categorical where appropriate
        elif col_type == "object":
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])

            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype("category")

    optimized_mem = df.memory_usage().sum() / 1024**2  # Convert bytes to MB

    if verbose:
        print(
            f"Memory usage: Before={initial_mem:.2f}MB -> After={optimized_mem:.2f}MB, "
            f"Decreased by {100 * (initial_mem - optimized_mem) / initial_mem:.1f}%"
        )

    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a CSV or Parquet file and optimizes its memory usage.

    Args:
        file_path (str): The path to the data file (CSV or Parquet).

    Returns:
        pd.DataFrame: The loaded and optimized DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a CSV or Parquet file."
        )

    # Optimize memory usage
    df_optimized = optimize_memory_usage(df)

    print("Data loaded successfully.")
    return df_optimized


def save_data(
    df: pd.DataFrame, file_name: str, file_directory: str, file_format: str = "parquet"
) -> None:
    """
    Saves a pandas DataFrame to a CSV or Parquet file with a timestamped filename.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_name (str): The base name of the output file.
        file_directory (str): The directory where the output file will be saved.
        file_format (str, optional): The format of the output file (CSV or Parquet). Defaults to "parquet".

    Raises:
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_directory):
        os.makedirs(file_directory, exist_ok=True)

    time_now = time.strftime("%Y%m%d-%H%M")
    filename = f"{time_now}-{file_name}.{file_format}"
    filepath = os.path.join(file_directory, filename)

    # Save file
    if file_format == "parquet":
        df.to_parquet(filepath, index=False)
    elif file_format == "csv":
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    print(f"Data saved successfully: {filepath}")
