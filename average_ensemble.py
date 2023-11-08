import pandas as pd

df1 = pd.read_csv('predictions_updated_150.csv')
df2 = pd.read_csv('predictions_updated_7.csv')

# Verify that the IDs match in both DataFrames (assuming they are sorted and have the same length)
if not df1['id'].equals(df2['id']):
    raise ValueError("The IDs in both CSV files do not match.")

# Calculate the average of the first 720 predictions
df_average_first_720 = pd.DataFrame({
    'id': df1['id'].iloc[:720],
    'prediction': (df1['prediction'].iloc[:720] + df2['prediction'].iloc[:720])/2
})

# Keep the middle predictions from df1 as is
df_middle = df1.iloc[720:-720].copy()

# Calculate the average of the last 720 predictions
df_average_last_720 = pd.DataFrame({
    'id': df1['id'].iloc[-720:],
    'prediction': (df1['prediction'].iloc[-720:] + df2['prediction'].iloc[-720:])/2
})

# Concatenate the three DataFrames
df_combined = pd.concat([df_average_first_720, df_middle, df_average_last_720], ignore_index=True)

# Save the combined predictions to a new CSV file
df_combined.to_csv('combined_predictions_150_7.csv', index=False)
