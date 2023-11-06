import pandas as pd
import numpy as np

def rank_features_by_correlation(data_parquet_file, observed_parquet_file = None, target_parquet_file = None, threshold=None, num_features_to_keep=None):
    # Load main dataset, observed dataset and target dataset
    df_data = pd.read_parquet(data_parquet_file).head(5000)
    if observed_parquet_file is not None:
        df_observed = pd.read_parquet(observed_parquet_file).head(5000)
    
        # Concatenate the estimated and observed data vertically
        df_data = pd.concat([df_data, df_observed], axis=0, ignore_index=True)

    df_target = pd.read_parquet(target_parquet_file)

    # Check if 'date_forecast' and 'time' columns exist in their respective DataFrames
    if 'date_forecast' not in df_data.columns or 'time' not in df_target.columns:
        raise ValueError("'date_forecast' or 'time' column not found in one or both datasets")

    # Merge the datasets based on 'date_forecast' and 'time'
    df_merged = pd.merge(df_data, df_target, left_on='date_forecast', right_on='time', how='inner')

    # Feature Engineering

    # 1. Time-based Features
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['weekday'] = df_merged['time'].dt.weekday

    # Sinusoidal transformations for cyclical time features
    df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged['hour'] / 24)
    df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged['hour'] / 24)
    df_merged['month_sin'] = np.sin(2 * np.pi * df_merged['month'] / 12)
    df_merged['month_cos'] = np.cos(2 * np.pi * df_merged['month'] / 12)c
    df_merged['weekday_sin'] = np.sin(2 * np.pi * df_merged['weekday'] / 7)
    df_merged['weekday_cos'] = np.cos(2 * np.pi * df_merged['weekday'] / 7)


    # 2. Interaction Features
    df_merged['snow_depth_radiation'] = df_merged['snow_depth:cm'] * df_merged['direct_rad:W']


    # 6. NaN Handling
    df_merged['snow_density:kgm3'].fillna(df_merged['snow_density:kgm3'].median(), inplace=True)
    df_merged['ceiling_height_agl:m'].fillna(df_merged['ceiling_height_agl:m'].median(), inplace=True)
    df_merged['cloud_base_agl:m'].fillna(df_merged['cloud_base_agl:m'].median(), inplace=True)

    # 7. Binning
    bins = [-90, 0, 45, 90]
    labels = ['Low', 'Medium', 'High']
    df_merged['sun_elevation_binned'] = pd.cut(df_merged['sun_elevation:d'], bins=bins, labels=labels, include_lowest=True)



    # 4. Derivative Features
    df_merged['rad_change'] = df_merged['direct_rad:W'].diff().fillna(0)
    # Similar calculations for other columns if deemed important.


    # One-hot encode 'sun_elevation_binned'
    df_merged = pd.get_dummies(df_merged, columns=['sun_elevation_binned'])

    # Drop Constant Features
    df_merged.drop(['elevation:m', 'snow_drift:idx', 'snow_melt_10min:mm', 'wind_speed_w_1000hPa:ms'], axis=1, inplace=True)
    df_merged = manipulate_single_feature(df_merged, 'clear_sky_rad:W')
    # Get the name of the target column (assuming it's the last column in df_target)
    target_column = df_target.columns[-1]
    
    # Calculate correlation of each feature with the target column
    correlation_scores = df_merged.drop(['date_forecast', 'time'], axis=1).corr()[target_column].drop(target_column)

    # Rank features based on the absolute value of their correlation scores
    ranked_features = correlation_scores.abs().sort_values(ascending=False)

    if threshold is not None:
        ranked_features = ranked_features[ranked_features > threshold]

    if num_features_to_keep is not None:
        ranked_features = ranked_features.head(num_features_to_keep)
    
    return ranked_features


def manipulate_single_feature(df, feature_name):
    """
    Manipulate a single feature: 
    - Removing outliers
    - Applying log transformation (if all values are positive)
    - Returns a DataFrame with the manipulated feature and potential new features.
    """
    
    # 1. Outlier Removal
    low, high = np.percentile(df[feature_name], [3, 97])
    median = df[feature_name].median()
    df[feature_name + '_no_outliers'] = np.where(df[feature_name] < low, median, df[feature_name])
    df[feature_name + '_no_outliers'] = np.where(df[feature_name] > high, median, df[feature_name])
    
    # 2. Log Transformation
    if df[feature_name].min() > 0:
        df[feature_name + '_log'] = np.log(df[feature_name])

    return df



# Usage example
data_parquet_file = "data/A/X_train_estimated.parquet"
observed_parquet_file = "data/A/X_train_observed.parquet"
target_parquet_file = "data/A/train_targets.parquet"

ranked_features = rank_features_by_correlation(data_parquet_file=data_parquet_file, observed_parquet_file=observed_parquet_file, target_parquet_file=target_parquet_file, threshold=0.1)
# ranked_features = rank_features_by_correlation(data_parquet_file, target_parquet_file=target_parquet_file, threshold=0.1)
ranked_features = rank_features_by_correlation(data_parquet_file, target_parquet_file=target_parquet_file, threshold=0.1)


print(ranked_features)

