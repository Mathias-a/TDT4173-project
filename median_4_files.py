import pandas as pd

# Load the datasets
df1 = pd.read_csv('predictions_updated_7.csv')
df2 = pd.read_csv('predictions_updated_8.csv')
df3 = pd.read_csv('predictions_updated_150.csv')
df4 = pd.read_csv('predictions_updated_155.csv')

# Verify that the IDs match in all DataFrames (assuming they are sorted and have the same length)
if not (df1['id'].equals(df2['id']) and df1['id'].equals(df3['id']) and df1['id'].equals(df4['id'])):
    raise ValueError("The IDs in the CSV files do not match.")

# Combine the prediction columns into a single DataFrame for ensemble
df_ensemble = pd.DataFrame({
    'id': df1['id'],
    'prediction1': df1['prediction'],
    'prediction2': df2['prediction'],
    'prediction3': df3['prediction'],
    'prediction4': df4['prediction']
})

# Calculate the median of the predictions for all rows
df_ensemble['prediction'] = df_ensemble[['prediction1', 'prediction2', 'prediction3', 'prediction4']].median(axis=1)

# Ensure there are no negative predictions and round down values under 1 to zero

# Save the median ensemble predictions to a new CSV file
df_ensemble[['id', 'prediction']].to_csv('median_ensemble_predictions.csv', index=False)
