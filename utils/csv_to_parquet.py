import pyarrow.csv as pv
import pyarrow.parquet as pq

def csv_to_parquet_pyarrow(csv_file_path, parquet_file_path):
    """
    Converts a CSV file to a Parquet file using pyarrow.

    Parameters:
        csv_file_path (str): Path to the input CSV file.
        parquet_file_path (str): Path to the output Parquet file.

    Returns:
        None: Saves the Parquet file to the specified path.
    """
    try:
        # Read CSV into a PyArrow Table
        table = pv.read_csv(csv_file_path)

        # Write the Table to a Parquet file
        pq.write_table(table, parquet_file_path)

        print(f"Successfully converted '{csv_file_path}' to '{parquet_file_path}'.")
    except Exception as e:
        print(f"An error occurred while converting CSV to Parquet: {e}")

if __name__ == '__main__':
    csv_file_path = 'data/sample.csv'
    # csv_file_path = 'data/raw/csv/test_unlabeled_data-1.csv'
    parquet_file_path = 'data/sample.parquet'
    # parquet_file_path = 'data/raw/parquet/test_unlabeled_data-1.parquet'
    csv_to_parquet_pyarrow(csv_file_path, parquet_file_path)
