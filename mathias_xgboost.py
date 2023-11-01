# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

# %% [markdown]
# # Hyperparameters

# %%

CUSTOM_COLUMNS_TO_KEEP = [
    # "hour_cos",
    # "month_sin",
    # "hour_sin",
    # "month_cos",
]

COLUMNS_TO_KEEP = [
    "direct_rad:W",
    "clear_sky_rad:W",
    "diffuse_rad:W",
    "direct_rad_1h:J",
    "is_in_shadow:idx",
    "clear_sky_energy_1h:J",
    "diffuse_rad_1h:J",
    "is_day:idx",
    "sun_elevation:d",
    "ceiling_height_agl:m",
    "effective_cloud_cover:p",
    "visibility:m",
    "total_cloud_cover:p",
    "air_density_2m:kgm3",
    "wind_speed_v_10m:ms",
    "dew_point_2m:K",
    "wind_speed_u_10m:ms",
    't_1000hPa:K',
    'absolute_humidity_2m:gm3',
    'snow_water:kgm2',
    'relative_humidity_1000hPa:p',
    'fresh_snow_24h:cm',
    'cloud_base_agl:m',
    'fresh_snow_12h:cm',
    'snow_depth:cm',
    'dew_or_rime:idx',
    'fresh_snow_6h:cm',
    'super_cooled_liquid_water:kgm2',
    'fresh_snow_3h:cm',
    'rain_water:kgm2',
    'precip_type_5min:idx',
    'precip_5min:mm',
    'fresh_snow_1h:cm',
    'sun_azimuth:d',
    'msl_pressure:hPa',
    'pressure_100m:hPa',
    'pressure_50m:hPa',
    'sfc_pressure:hPa',
    'prob_rime:p',
    'wind_speed_10m:ms',
    'elevation:m',
    'snow_density:kgm3',
    'snow_drift:idx',
    'snow_melt_10min:mm',
    'wind_speed_w_1000hPa:ms',
    # "date_calc",
    "pv_measurement",
] + CUSTOM_COLUMNS_TO_KEEP

LOCATION = "A"

# %% [markdown]
# # Load Data

# %%
# 1. Load data
df_observed = pd.read_parquet(f"data/{LOCATION}/X_train_observed.parquet")
df_estimated = pd.read_parquet(f"data/{LOCATION}/X_train_estimated.parquet")
df_target = pd.read_parquet(f"data/{LOCATION}/train_targets.parquet")

# 2. Combine observed and estimated datasets
df_combined = pd.concat([df_observed, df_estimated], axis=0).sort_values(by="date_forecast")

# 3. Merge with target data
df_merged = pd.merge(df_combined, df_target, left_on="date_forecast", right_on="time", how="inner")

# %% [markdown]
# # Downsampling and Feature Engineering

# %%
# Downsampling the dataframe to hourly intervals
df_merged = df_merged.resample("H", on="date_forecast").mean()

# Keep only relevant columns
df_merged = df_merged[COLUMNS_TO_KEEP]

# 4. Extract features and target
df_merged = df_merged.dropna(subset=["pv_measurement"])
df_merged.fillna(0, inplace=True)  # Fill NaN values
y = df_merged["pv_measurement"]
X = df_merged.drop("pv_measurement", axis=1)

# %% [markdown]
# # Split Data into Train and Validation

# %%
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# %% [markdown]
# # Baseline model

# %%


def naive_forecast(time_series, steps=1):
    """
    This function returns the time_series shifted by a given number of steps.
    For our purpose, we'll use 24-hour shift.
    """
    return time_series.shift(steps)


# Calculate the predictions using naive forecast
baseline_predictions = naive_forecast(y_train, steps=24)

# Only consider the predictions where both the actual and predicted values are available
mask = (~baseline_predictions.isna()) & (~y_train.isna())

# Calculate MAE
baseline_mae = np.mean(np.abs(baseline_predictions[mask] - y_train[mask]))
print(f"Baseline MAE: {baseline_mae}")

# %% [markdown]
# # XGBoost Model

# %%

# Convert data to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'min_child_weight': 5
}

# Train XGBoost model
num_rounds = 1000
xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=[(dtrain, 'train'), (dval, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=10
)

# %% [markdown]
# # XGBoost Prediction and Evaluation

# %%

# Predict with XGBoost
y_pred_xgb = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration + 1))

# Calculate MAE for XGBoost
xgb_mae = np.mean(np.abs(y_pred_xgb - y_val))
print(f"XGBoost MAE: {xgb_mae}")

xgb.plot_importance(xgb_model)
plt.show()





# %%