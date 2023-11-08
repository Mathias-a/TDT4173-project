import pandas as pd

df1 = pd.read_csv('predictions_updated_150.csv')
df2 = pd.read_csv('predictions_updated_7.csv')

# Verify that the IDs match in both DataFrames (assuming they are sorted and have the same length)
if not df1['id'].equals(df2['id']):
    raise ValueError("The IDs in both CSV files do not match.")

# Calculate the average of the predictions for all rows
df_average_all = pd.DataFrame({
    'id': df1['id'],
    'prediction': (df1['prediction'] + df2['prediction']) / 2
})

# Save the averaged predictions to a new CSV file
df_average_all.to_csv('averaged_predictions_all_7_150.csv', index=False)