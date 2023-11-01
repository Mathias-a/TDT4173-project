# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.diagnostics import cross_validation
from itertools import product
import xgboost as xgb

# %% [markdown]
# # Hyperparameters
#

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
    # "total_cloud_cover:p",
    "air_density_2m:kgm3",
    "wind_speed_v_10m:ms",
    "dew_point_2m:K",
    # "wind_speed_u_10m:ms",
    # 't_1000hPa:K',
    # 'absolute_humidity_2m:gm3',
    # 'snow_water:kgm2',
    # 'relative_humidity_1000hPa:p',
    # 'fresh_snow_24h:cm',
    # 'cloud_base_agl:m',
    # 'fresh_snow_12h:cm',
    # 'snow_depth:cm',
    # 'dew_or_rime:idx',
    # 'fresh_snow_6h:cm',
    # 'super_cooled_liquid_water:kgm2',
    # 'fresh_snow_3h:cm',
    # 'rain_water:kgm2',
    # 'precip_type_5min:idx',
    # 'precip_5min:mm',
    # 'fresh_snow_1h:cm',
    # 'sun_azimuth:d',
    # 'msl_pressure:hPa',
    # 'pressure_100m:hPa',
    # 'pressure_50m:hPa',
    # 'sfc_pressure:hPa',
    # 'prob_rime:p',
    # 'wind_speed_10m:ms',
    # 'elevation:m',
    # 'snow_density:kgm3',
    # 'snow_drift:idx',
    # 'snow_melt_10min:mm',
    # 'wind_speed_w_1000hPa:ms',
    # "date_calc",
    "pv_measurement",
] + CUSTOM_COLUMNS_TO_KEEP

LEARNING_RATE = 0.00008
NUM_EPOCHS = 200
BATCH_SIZE = 32
NUM_FEATURES = (
    len(COLUMNS_TO_KEEP) - 1
)  # -1 because pv_measurement is the target, +4 FOR HOUR
FEATURE_SIZE = 4  # 7 days of hourly data
WEIGHT_DECAY = 0.12
SEQUENCE_LENGTH = 24 * 2
LOCATION = "A"

# %% [markdown]
# # Neural net
#


# %%
def create_sequences(data, sequence_length):
    """
    Converts time series data into overlapping sequences/windows.
    """
    sequences = []
    target_length = 1
    for i in range(len(data) - sequence_length + 1):
        seq = data[i : i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)


def apply_pca(data, q=None, center=True, niter=2):
    """
    Applies PCA to the given data tensor.
    Returns transformed data and PCA components.
    """
    U, S, V = torch.pca_lowrank(data, q=q, center=center, niter=niter)
    transformed_data = torch.matmul(data, V)
    return transformed_data, V


class SolarPredictionNet(nn.Module):
    def __init__(self, sequence_length, num_channels):
        super(SolarPredictionNet, self).__init__()

        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=sequence_length, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Use dropout after first fully connected layer
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# %% [markdown]
# # Load and Preprocess Data

# %%
# 1. Load data
df_observed = pd.read_parquet(f"data/{LOCATION}/X_train_observed.parquet")
df_estimated = pd.read_parquet(f"data/{LOCATION}/X_train_estimated.parquet")
df_target = pd.read_parquet(f"data/{LOCATION}/train_targets.parquet")

# 2. Combine observed and estimated datasets
df_combined = pd.concat([df_observed, df_estimated], axis=0).sort_values(
    by="date_forecast"
)

# 3. Merge with target data
df_merged = pd.merge(
    df_combined, df_target, left_on="date_forecast", right_on="time", how="inner"
)

# %% [markdown]
# # Downsampling and Feature Engineering

# %%
# Downsampling the dataframe to hourly intervals
df_merged = df_merged.resample("H", on="date_forecast").mean()

# Add columns for hour of day, and month of year using sine and cosine to capture the cyclical nature
df_merged["hour_sin"] = np.sin(2 * np.pi * df_merged.index.hour / 24)
df_merged["hour_cos"] = np.cos(2 * np.pi * df_merged.index.hour / 24)

df_merged["month_sin"] = np.sin(2 * np.pi * df_merged.index.month / 12)
df_merged["month_cos"] = np.cos(2 * np.pi * df_merged.index.month / 12)

# Keep only relevant columns
df_merged = df_merged[COLUMNS_TO_KEEP]

# 4. Extract features and target
df_merged = df_merged.dropna(subset=["pv_measurement"])
df_merged.fillna(0, inplace=True)  # Fill NaN values
y = df_merged["pv_measurement"]
X = df_merged.drop("pv_measurement", axis=1)

# %% [markdown]
# # Data Preparation for Modeling

# %%
# Convert dataframes to sequences
X_sequences = create_sequences(X.values, SEQUENCE_LENGTH)
y_sequences = y.values[SEQUENCE_LENGTH - 1 :]  # Aligned with the end of each sequence

# Sequential Split
train_size = int(0.8 * len(X_sequences))
X_train, X_val = X_sequences[:train_size], X_sequences[train_size:]
y_train, y_val = y_sequences[:train_size], y_sequences[train_size:]

# Normalize the data and apply PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
    X_train.shape
)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

# Convert to tensors and apply PCA
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

X_train_tensor, pca_components = apply_pca(X_train_tensor, q=NUM_FEATURES)
X_val_tensor, _ = apply_pca(X_val_tensor, q=NUM_FEATURES)

# Ensure shapes are [batch, channels, sequence]
X_train_tensor = X_train_tensor.transpose(1, 2)
X_val_tensor = X_val_tensor.transpose(1, 2)

# Update NUM_FEATURES to the size after PCA
NUM_FEATURES = pca_components.shape[1]

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).transpose(1, 2)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# %% [markdown]
# # Dataset and DataLoader Preparation


# %%
class SolarDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create datasets for training and validation
train_dataset = SolarDataset(X_train_tensor, y_train_tensor)
val_dataset = SolarDataset(X_val_tensor, y_val_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %% [markdown]
# # Baseline model


# %%
# Baseline Model
def naive_forecast(time_series, steps=1):
    """
    This function returns the time_series shifted by a given number of steps.
    For our purpose, we'll shift by 365 days, and if data is not available,
    then we'll shift by 24 hours as a fallback.
    """
    return time_series.shift(steps)


# Calculate the predictions using naive forecast
baseline_predictions = naive_forecast(df_merged["pv_measurement"], steps=365 * 24)
if baseline_predictions.isna().sum() > 0:
    # Fallback to 24-hour shift if there are any NaN values
    baseline_predictions = naive_forecast(df_merged["pv_measurement"], steps=24)

# Only consider the predictions where both the actual and predicted values are available
mask = (~baseline_predictions.isna()) & (~df_merged["pv_measurement"].isna())

# Calculate MAE
baseline_mae = np.mean(
    np.abs(baseline_predictions[mask] - df_merged["pv_measurement"][mask])
)
print(f"Baseline MAE: {baseline_mae}")


# %% [markdown]
# # Prophet Model

# %%
# 1. Restructure df merged so it fits Prophet's requirements
prophet_dataset = df_merged.reset_index()
prophet_dataset = prophet_dataset.rename(
    columns={"date_forecast": "ds", "pv_measurement": "y"}
)
# Convert the 'ds' column to the desired format
prophet_dataset["ds"] = prophet_dataset["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Sort and split into train and validation
prophet_dataset.sort_values(by="ds", inplace=True)
prophet_dataset["floor"] = 0
prophet_dataset["cap"] = 6000
prophet_train = prophet_dataset[:train_size]
prophet_val = prophet_dataset[train_size:]


# 2. Create Prophet model
model = Prophet(
    growth="linear",
    changepoints=None,
    n_changepoints=0,
    changepoint_range=0,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    holidays=None,
    seasonality_mode="additive",
    seasonality_prior_scale=0.0001,
    holidays_prior_scale=0.01,
    changepoint_prior_scale=0.03,
    mcmc_samples=0,
    interval_width=0.8,
    uncertainty_samples=1000,
)

# Add regressors
for col in prophet_train.columns:
    if col not in ["ds", "y", "floor", "cap"]:
        model.add_regressor(col)

# 3. Fit model
model.fit(prophet_train)

# 4. Make predictions
prophet_predictions = model.predict(prophet_val)
prophet_predictions["yhat_orig"] = np.expm1(prophet_predictions["yhat"])


# 5. Calculate MAE
prophet_mae = np.mean(np.abs(prophet_predictions["yhat"].values - prophet_val["y"].values))

print(f"Prophet MAE in log scale: {prophet_mae}")

fig = model.plot(prophet_predictions)
fig.show()

# %%
param_grid = {
    'seasonality_prior_scale': [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0],
}
best_parameters = None
best_mae = float('inf')

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

for params in all_params:
    temp_model = Prophet(
        growth="linear",
        changepoints=None,
        n_changepoints=0,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        # seasonality_prior_scale=15.0,
        holidays_prior_scale=0.01,
        changepoint_prior_scale=0.03,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
    )

    # Add regressors
    for col in prophet_train.columns:
        if col not in ["ds", "y", "floor", "cap"]:
            temp_model.add_regressor(col)

    temp_model.fit(prophet_train)

    df_cv = cross_validation(temp_model, '365 days', initial='730 days', period='180 days')
    mae = np.mean(np.abs(df_cv["yhat"] - df_cv["y"]))
    if mae < best_mae:
        best_mae = mae
        best_parameters = params

print(f"Best MAE: {best_mae}")
print(best_parameters)

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
watchlist = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(xgb_params, dtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=10)

# %% [markdown]
# # XGBoost Prediction and Evaluation

# %%

# Predict with XGBoost
y_pred_xgb = xgb_model.predict(dval, ntree_limit=xgb_model.best_ntree_limit)

# Calculate MAE for XGBoost
xgb_mae = np.mean(np.abs(y_pred_xgb - y_val))
print(f"XGBoost MAE: {xgb_mae}")

# %%
