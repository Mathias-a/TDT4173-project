import pandas as pd

df1 = pd.read_csv('predictions_updated_150.csv')
df2 = pd.read_csv('predictions_updated_10.csv')
df3 = pd.read_csv('predictions_updated_11.csv')

# Verify that the IDs match in both DataFrames (assuming they are sorted and have the same length)


# Calculate the average of the predictions for all rows
df_average_all = pd.DataFrame({
    'id': df1['id'],
    'prediction': (df1['prediction']*0.25 + df2['prediction']*0.5 + df3['prediction']*0.25) 
})

# Save the averaged predictions to a new CSV file
df_average_all.to_csv('averaged_predictions_three best.csv', index=False)