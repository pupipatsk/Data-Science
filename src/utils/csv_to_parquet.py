import os
import pyarrow
import pyarrow.csv as pv
import pyarrow.parquet as pq


def csv_to_parquet_pyarrow(csv_file_path: str, parquet_file_path: str) -> None:
    """
    Converts a CSV file to a Parquet file using PyArrow.

    Args:
        csv_file_path (str): Path to the input CSV file.
        parquet_file_path (str): Path to the output Parquet file.

    Returns:
        None: Saves the Parquet file to the specified path.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        OSError: If there's an issue creating the output directory.
        pyarrow.lib.ArrowException: If an error occurs during conversion.
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(parquet_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Read CSV into a PyArrow Table
        table = pv.read_csv(csv_file_path)

        # Write the Table to a Parquet file
        pq.write_table(table, parquet_file_path)

        print(f"Successfully converted '{csv_file_path}' to '{parquet_file_path}'.")
    except (OSError, pyarrow.lib.ArrowException) as e:
        print(f"An error occurred while converting CSV to Parquet: {e}")
        raise


if __name__ == "__main__":
    csv_file_path = "path/to/input.csv"
    parquet_file_path = "path/to/output.parquet"
    csv_to_parquet_pyarrow(csv_file_path, parquet_file_path)
