import pandas as pd

def rank_features_by_correlation(data_parquet_file, target_parquet_file):
    # Load main dataset and target dataset
    df_data = pd.read_parquet(data_parquet_file).head(5000)
    df_target = pd.read_parquet(target_parquet_file)

    # Check if 'date_forecast' and 'time' columns exist in their respective DataFrames
    if 'date_forecast' not in df_data.columns or 'time' not in df_target.columns:
        raise ValueError("'date_forecast' or 'time' column not found in one or both datasets")

    # Merge the datasets based on 'date_forecast' and 'time'
    df_merged = pd.merge(df_data, df_target, left_on='date_forecast', right_on='time', how='inner')

    # Get the name of the target column (assuming it's the last column in df_target)
    target_column = df_target.columns[-1]

    # Check if target column exists in merged DataFrame
    if target_column not in df_merged.columns:
        raise ValueError(f"Target column '{target_column}' not found in merged dataset")

    # Calculate correlation of each feature with the target column
    correlation_scores = df_merged.drop(['date_forecast', 'time'], axis=1).corr()[target_column].drop(target_column)

    # Rank features based on the absolute value of their correlation scores
    ranked_features = correlation_scores.abs().sort_values(ascending=False)

    return ranked_features

# Usage example
data_parquet_file = "data-2/A/X_train_estimated.parquet"
target_parquet_file = "data-2/A/train_targets.parquet"
ranked_features = rank_features_by_correlation(data_parquet_file, target_parquet_file)
print(ranked_features)
