import pyarrow.parquet as pq

def print_column_names_with_pyarrow(parquet_file_path):
    # Read the Parquet file's metadata
    parquet_file = pq.ParquetFile(parquet_file_path)
    
    # Print column names
    for column in parquet_file.schema.names:
        print(column)

# Usage
file_path = "data-2/A/train_targets.parquet"
print_column_names_with_pyarrow(file_path)
