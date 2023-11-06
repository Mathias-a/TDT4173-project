import pandas as pd
import numpy as np

def generate_and_save_engineered_features(estimated_parquet_file=None, output_parquet_file="engineered_data.parquet"):
    # Load main dataset, observed dataset, and target dataset
    df_merged= pd.read_parquet(estimated_parquet_file)

    print(df_merged.head())


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


def manipulate_single_feature(df, feature_name):
    # 1. Outlier Removal
    # ... this function remains the same ...

    return df


# Usage example
estimated_parquet_file = "data/B/X_test_estimated.parquet"
output_file = "cleaned_data/B/X_test_engineered.parquet"

generate_and_save_engineered_features(estimated_parquet_file=estimated_parquet_file, output_parquet_file=output_file)
