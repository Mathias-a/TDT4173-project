# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

# %% [markdown]
# # Hyperparameters
# 

# %%

CUSTOM_COLUMNS_TO_KEEP = [
    "hour_cos",
    # "month_sin",
    "hour_sin",
    # "month_cos",
]

COLUMNS_TO_KEEP = [
    "direct_rad:W",
    "clear_sky_rad:W",
    "diffuse_rad:W",
    # "direct_rad_1h:J",
    "is_in_shadow:idx",
    "clear_sky_energy_1h:J",
    # "diffuse_rad_1h:J",
    "is_day:idx",
    "sun_elevation:d",
    # "ceiling_height_agl:m",
    # "effective_cloud_cover:p",
    "visibility:m",
    'total_cloud_cover:p',
    # 'air_density_2m:kgm3',
    # 'wind_speed_v_10m:ms',
    # 'dew_point_2m:K',
    # 'wind_speed_u_10m:ms',
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
NUM_FEATURES = len(COLUMNS_TO_KEEP) - 1  # -1 because pv_measurement is the target, +4 FOR HOUR
FEATURE_SIZE = 4  # 7 days of hourly data
WEIGHT_DECAY = 0.12
SEQUENCE_LENGTH = 24*2
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
df_combined = pd.concat([df_observed, df_estimated], axis=0).sort_values(by='date_forecast')

# 3. Merge with target data
df_merged = pd.merge(
    df_combined, df_target, left_on="date_forecast", right_on="time", how="inner"
)

# %% [markdown]
# # Downsampling and Feature Engineering

# %%
# Downsampling the dataframe to hourly intervals
df_merged = df_merged.resample('H', on="date_forecast").mean()

# Add columns for hour of day, and month of year using sine and cosine to capture the cyclical nature
df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged.index.hour / 24)
df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged.index.hour / 24)

df_merged['month_sin'] = np.sin(2 * np.pi * df_merged.index.month / 12)
df_merged['month_cos'] = np.cos(2 * np.pi * df_merged.index.month / 12)

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
y_sequences = y.values[SEQUENCE_LENGTH-1:]  # Aligned with the end of each sequence

# Sequential Split
train_size = int(0.8 * len(X_sequences))
X_train, X_val = X_sequences[:train_size], X_sequences[train_size:]
y_train, y_val = y_sequences[:train_size], y_sequences[train_size:]

# Normalize the data and apply PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
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
# # Training Loop
# 

# %%
def train_model(model, train_loader, val_loader):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # Reduce LR by a factor of 0.1 every 50 epochs

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_train_loss = 0.0  # Initialize accumulated training loss

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_train_loss += loss.item()  # Accumulate the training loss
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)  # Compute the average training loss

        # Evaluate the model on the validation set (This part remains unchanged)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                output = torch.where(output < 0, torch.zeros_like(output), output)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)  # Compute the average validation loss

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}"
        )
        
        scheduler.step()  # Step the learning rate scheduler

    print("Training complete!")




# %%
def save_model(model, location):
    filename = f"model_location_{location}.pt"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# ... [Training Loop]

model = SolarPredictionNet(SEQUENCE_LENGTH, NUM_FEATURES)
train_model(model, train_loader, val_loader)
save_model(model, LOCATION) 

# %%

def load_model(location):
    model = SolarPredictionNet(SEQUENCE_LENGTH, NUM_FEATURES)
    model.load_state_dict(torch.load(f"mathias_location_{location}.pt"))
    model.eval()
    return model

def pad_data(data, sequence_length):
    """
    Pads the data with zeros at the beginning to ensure 
    the final number of sequences matches the original number of data points.
    """
    padding = np.zeros((sequence_length - 1, data.shape[1]))
    return np.vstack((padding, data))


def make_predictions(location, df_test):
    # Load model
    model = load_model(location)
    
    # Ensure the index is a datetime
    df_test['date_forecast'] = pd.to_datetime(df_test['date_forecast'])
    # Set the date_calc column as the index for resampling
    df_test.set_index('date_forecast', inplace=True)
    # Resample to 1-hour intervals
    df_test = df_test.resample('1H').mean()
    df_test = df_test.dropna(how='all').reset_index(drop=True)
    
    # Keep only the columns used during training (minus the target column)
    relevant_columns = [col for col in COLUMNS_TO_KEEP if col != "pv_measurement"]
    df_test = df_test[relevant_columns]
    
    # Fill NaNs (if any after resampling)
    df_test.fillna(0, inplace=True)
    # Create sequences and normalize
    padded_data = pad_data(df_test.values, SEQUENCE_LENGTH)
    test_sequences = create_sequences(padded_data, SEQUENCE_LENGTH)
    test_sequences = scaler.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)
    test_tensor = torch.tensor(test_sequences, dtype=torch.float32).transpose(1, 2)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(test_tensor)
        predictions = predictions.numpy().flatten()
    return predictions



# Read the Kaggle test.csv to get the location and ids
df_submission = pd.read_csv("data/test.csv")

locations = ["A", "B", "C"]

# Iterate over the locations and fill in the predictions
for loc in locations:
    print(loc)
    # Load forecasted weather data for testing for the current location
    df_loc = pd.read_parquet(f"data/{loc}/X_test_estimated.parquet")
    preds = make_predictions(loc, df_loc)
    # Assign the predictions to df_submission for the current location
    mask = df_submission["location"] == loc
    df_submission.loc[mask, "prediction"] = preds

# Save the results to a new submission file
df_submission[["id", "prediction"]].to_csv("sample_kaggle_submission.csv", index=False)



