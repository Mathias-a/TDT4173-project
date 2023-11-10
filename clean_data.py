import pandas as pd
import numpy as np

def generate_and_save_engineered_features(estimated_parquet_file=None, observed_parquet_file=None, target_parquet_file=None, output_parquet_file="engineered_data.parquet"):
    # Load main dataset, observed dataset, and target dataset
    df_estimated = pd.read_parquet(estimated_parquet_file)
    df_observed = pd.read_parquet(observed_parquet_file)

    df_target = pd.read_parquet(target_parquet_file)
    df_merged = pd.concat([df_observed, df_estimated], axis=0, ignore_index=True)
    print(df_merged.head())

    # Check if 'date_forecast' and 'time' columns exist in their respective DataFrames
    if 'date_forecast' not in df_merged.columns or 'time' not in df_target.columns:
        raise ValueError("'date_forecast' or 'time' column not found in one or both datasets")


    # Feature Engineering
    df_merged = feature_engineering(df_merged)

    # Save the engineered data
    df_merged.to_parquet(output_parquet_file)
    print(f"Engineered data saved to {output_parquet_file}")


def feature_engineering(df_merged):
    # 1. Time-based Features
    df_merged['hour'] = df_merged['date_forecast'].dt.hour
    df_merged['month'] = df_merged['date_forecast'].dt.month
    df_merged['weekday'] = df_merged['date_forecast'].dt.weekday

    # Sinusoidal transformations for cyclical time features
    df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged['hour'] / 24)
    df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged['hour'] / 24)
    df_merged['month_sin'] = np.sin(2 * np.pi * df_merged['month'] / 12)
    df_merged['month_cos'] = np.cos(2 * np.pi * df_merged['month'] / 12)
    df_merged['weekday_sin'] = np.sin(2 * np.pi * df_merged['weekday'] / 7)
    df_merged['weekday_cos'] = np.cos(2 * np.pi * df_merged['weekday'] / 7)

    # Binning
    bins = [-90, 0, 45, 90]
    labels = ['1', '2', '3']
    df_merged['sun_elevation_binned'] = pd.cut(df_merged['sun_elevation:d'], bins=bins, labels=labels, include_lowest=True)


    return df_merged




# Usage example
estimated_parquet_file = "data/C/X_train_estimated.parquet"
observed_parquet_file = "data/C/X_train_observed.parquet"
target_parquet_file = "data/C/train_targets.parquet"
output_file = "cleaned_data/C/X_train_engineered.parquet"

generate_and_save_engineered_features(estimated_parquet_file=estimated_parquet_file, observed_parquet_file=observed_parquet_file, target_parquet_file=target_parquet_file, output_parquet_file=output_file)
