import pandas as pd

df_observed = pd.read_parquet(f"data/A/X_train_observed.parquet")
df_estimated= pd.read_parquet(f"data/A/X_train_estimated.parquet")
df_target = pd.read_parquet(f"data/A/train_targets.parquet")
df_test = pd.read_parquet(f"data/A/X_test_estimated.parquet")

df_observed.to_csv(f"data/A/CSV/X_train_observed_A.csv", index=False)
df_estimated.to_csv(f"data/A/CSV/X_train_estimated_A.csv", index=False)
df_target.to_csv(f"data/A/CSV/train_targets_A.csv", index=False)
df_test.to_csv(f"data/A/CSV/X_test_estimated_A.csv", index=False)

