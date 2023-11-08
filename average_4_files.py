import pandas as pd

# Load the datasets
df1 = pd.read_csv('predictions_updated_7.csv')
df2 = pd.read_csv('predictions_updated_8.csv')
df3 = pd.read_csv('predictions_updated_150.csv')
df4 = pd.read_csv('predictions_updated_155.csv')

# Verify that the IDs match in all DataFrames (assuming they are sorted and have the same length)
if not (df1['id'].equals(df2['id']) and df1['id'].equals(df3['id']) and df1['id'].equals(df4['id'])):
    raise ValueError("The IDs in the CSV files do not match.")

# Calculate the average of the predictions for all rows
df_average = pd.DataFrame({
    'id': df1['id'],
    'prediction_average': (df1['prediction'] + df2['prediction'] + df3['prediction'] + df4['prediction']) / 4
})

# Save the averaged predictions to a new CSV file
df_average.to_csv('averaged_predictions.csv', index=False)