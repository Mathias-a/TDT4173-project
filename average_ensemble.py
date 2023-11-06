import pandas as pd

df1 = pd.read_csv('predictions_updated_3.csv')
df2 = pd.read_csv('predictions_updated_155.csv')

# Verify that the IDs match in both DataFrames (assuming they are sorted and have the same length)
if not df1['id'].equals(df2['id']):
    raise ValueError("The IDs in both CSV files do not match.")

# Calculate the average of the predictions
df_average = pd.DataFrame({
    'id': df1['id'],
    'prediction': (df1['prediction']*0.25 + df2['prediction']*0.75)
})

# Save the averaged predictions to a new CSV file
df_average.to_csv('averaged_predictions.csv', index=False)
