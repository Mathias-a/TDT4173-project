# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


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
    "t_1000hPa:K",
    "absolute_humidity_2m:gm3",
    "snow_water:kgm2",
    "relative_humidity_1000hPa:p",
    "fresh_snow_24h:cm",
    "cloud_base_agl:m",
    "fresh_snow_12h:cm",
    "snow_depth:cm",
    "dew_or_rime:idx",
    "fresh_snow_6h:cm",
    "super_cooled_liquid_water:kgm2",
    "fresh_snow_3h:cm",
    "rain_water:kgm2",
    "precip_type_5min:idx",
    "precip_5min:mm",
    "fresh_snow_1h:cm",
    "sun_azimuth:d",
    "msl_pressure:hPa",
    "pressure_100m:hPa",
    "pressure_50m:hPa",
    "sfc_pressure:hPa",
    "prob_rime:p",
    "wind_speed_10m:ms",
    "elevation:m",
    "snow_density:kgm3",
    "snow_drift:idx",
    "snow_melt_10min:mm",
    "wind_speed_w_1000hPa:ms",
    # "date_calc",
    "pv_measurement",
] + CUSTOM_COLUMNS_TO_KEEP

LOCATION = "A"
SHIFTS = [24 * 31, 24*62, 7*24]
MODEL_FILENAME = f'models/xgboost_model_{LOCATION}.json'

# %% [markdown]
# # Load Data

# %%
# 1. Load data
df_observed = pd.read_parquet(f"data/{LOCATION}/X_train_observed.parquet")
df_estimated = pd.read_parquet(f"data/{LOCATION}/X_train_estimated.parquet")
df_target = pd.read_parquet(f"data/{LOCATION}/train_targets.parquet")

# 2. Combine observed and estimated datasets
df_combined = pd.concat([df_observed, df_estimated], axis=0).sort_values(
    by="date_forecast"
)

df_merged = df_combined.resample("H", on="date_forecast").mean()

# 3. Merge with target data
df_merged = pd.merge(
    df_combined, df_target, left_on="date_forecast", right_on="time", how="inner"
)
# %% [markdown]
# # Downsampling and Feature Engineering

# %%
# Downsampling the dataframe to hourly intervals

# Keep only relevant columns
df_merged = df_merged[COLUMNS_TO_KEEP]

# 4. Extract features and target
df_merged = df_merged.dropna(subset=["pv_measurement"])
df_merged.fillna(0, inplace=True)  # Fill NaN values

# %% [markdown]
# # Add lagged features


# %%
def add_lagged_features(df, features, shift_value):
    """
    This function takes in a dataframe, a list of features, and a shift interval.
    It returns the dataframe with the lagged features added.
    """
    for feature in features:
        df[f"{feature}_lagged_{shift_value}h"] = df[feature].shift(shift_value)
    return df


for shift in SHIFTS:
    df_merged = add_lagged_features(df_merged, COLUMNS_TO_KEEP, shift)

# Remember to drop NaN values introduced by shifting
df_merged.dropna(inplace=True)


# %% [markdown]
# # Remove outliers


# %%
# Remove outliers
def remove_outliers(df):
    """
    Removes outliers in a dataframe based on IQR for each column.

    Parameters:
    - df (DataFrame): input dataframe

    Returns:
    - DataFrame: dataframe with outliers removed
    """
    Q1 = df.quantile(0.05)
    Q3 = df.quantile(0.95)
    IQR = Q3 - Q1
    outlier_condition = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    return df[outlier_condition].dropna()


# Remove outliers from the merged dataset
# df_merged = remove_outliers(df_merged)


# %% [markdown]
# # Split Data into Train and Validation

# %%
# Split data
y = df_merged["pv_measurement"]
X = df_merged.drop("pv_measurement", axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

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

# Modified XGBoost parameters to prevent overfitting
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "eta": 0.0005,  # Reduced learning rate
    "max_depth": 7,  # Reduced tree depth
    "subsample": 0.7,  # Reduced subsampling
    "colsample_bytree": 0.8,  # Reduced column sampling
    "min_child_weight": 7,  # Increased min_child_weight
    "alpha": 0.4,  # L1 regularization
    "lambda": 1,  # L2 regularization
}

# Train XGBoost model with modifications
num_rounds = 50000  # Increased boosting rounds due to reduced eta
xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=[(dtrain, "train"), (dval, "eval")],
    early_stopping_rounds=200,  # Increased early stopping rounds
    verbose_eval=10,
)
xgb_model.save_model(MODEL_FILENAME)


# %% [markdown]
# # XGBoost Prediction and Evaluation

# %%

xgb_loaded_model = xgb.Booster()
xgb_loaded_model.load_model(MODEL_FILENAME)

# Predict with XGBoost
y_pred_xgb = xgb_loaded_model.predict(dval, iteration_range=(0, xgb_loaded_model.best_iteration + 1))

# Calculate MAE for XGBoost
xgb_mae = np.mean(np.abs(y_pred_xgb - y_val))
print(f"XGBoost MAE: {xgb_mae}")

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_val.reset_index(drop=True), label="Actual", color='blue')
plt.plot(y_pred_xgb, label="Predicted", color='red')
plt.title("XGBoost Predictions vs Actuals")
plt.legend()
plt.show()

xgb.plot_importance(xgb_loaded_model, max_num_features=10)
plt.show()


# %% [markdown]
# # Predict on test set


# %%



# %% [markdown]
# # XGBoost finding opptimal hyperparameters


# %%
# Define the hyperparameter space
def optimize_xgb():
    param_grid = {
        "objective": ["reg:squarederror"],
        "eval_metric": ["mae"],
        "eta": [
            0.001,  # Best value I
            0.005,
            0.01,
            # 0.05,
            # 0.1,
            # 0.3,
        ],
        "max_depth": [
            # 3,
            # 4,
            # 5,
            6,
            7,  # Best value II
            8,
            # 9,
            # 10,
            # 12,
        ],
        "subsample": [
            # 0.3,
            # 0.4,
            # 0.5,
            # 0.6,
            0.7,  # Best value II
            0.8,  # Best value II
            0.9,
            # 1.0,
        ],
        "colsample_bytree": [
            # 0.3,
            # 0.4,
            # 0.5,
            # 0.6,
            0.7,
            0.8,  # Best value II
            0.9,  # Best value II
            # 1.0,
        ],
        "min_child_weight": [
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            6,
            7,  # Best value III
            8,  # Best value I
            # 9,
            # 10,
            # 11,
            # 12,
        ],
    }

    # Initialize XGBoost Regressor
    xgb_optimization_model = xgb.XGBModel(
        learning_rate=0.02,
        n_estimators=600,
        objective="reg:squarederror",
        silent=True,
        nthread=1,
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        xgb_optimization_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=None,
        verbose=0,  # make it silent
        n_jobs=-1,
    )

    # Fit the model with early stopping rounds and validation data
    fit_params = {
        "early_stopping_rounds": 50,
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }

    grid_search.fit(X_train, y_train, **fit_params)

    # Print best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Predict on validation set
    y_pred_optimized = grid_search.predict(X_val)

    # Calculate MAE for optimized XGBoost
    mae_optimized = np.mean(np.abs(y_pred_optimized - y_val))
    print(f"Optimized XGBoost MAE: {mae_optimized}")


# %%
