import pandas as pd

df1 = pd.read_csv('predictions_updated_12.csv')
df2 = pd.read_csv('predictions_updated_11.csv')
df3 = pd.read_csv('predictions_updated_10.csv')
df4 = pd.read_csv('predictions_updated_13.csv')

# Verify that the IDs match in both DataFrames (assuming they are sorted and have the same length)


# Calculate the average of the first 720 predictions
df_average_first_720 = pd.DataFrame({
    'id': df1['id'].iloc[:720],
    'prediction': (df1['prediction'].iloc[:720] + df3['prediction'].iloc[:720]) / 2
})

df_average_middle_720 = pd.DataFrame({
    'id': df1['id'].iloc[720:1440],
    'prediction': (df4['prediction'].iloc[720:1440] *0.1 + df3['prediction'].iloc[720:1440]*0.1 + df2['prediction'].iloc[720:1440]*0.8)
})
# Keep the middle predictions from df1 as is
df_average_last = pd.DataFrame({
    'id': df1['id'].iloc[1440:],
    'prediction': (df2['prediction'].iloc[1440:] + df3['prediction'].iloc[1440:])/2
})


# Concatenate the three DataFrames
df_combined = pd.concat([df_average_first_720, df_average_middle_720, df_average_last], ignore_index=True)

# Save the combined predictions to a new CSV file
df_combined.to_csv('combined_predictions_A_B_C_weighted.csv', index=False)
