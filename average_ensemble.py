import pandas as pd

df1 = pd.read_csv('predictions_updated_10.csv')
df2 = pd.read_csv('autogluon_attempt_3.csv')
# Verify that the IDs match in both DataFrames (assuming they are sorted and have the same length)
if not df1['id'].equals(df2['id']):
    raise ValueError("The IDs in both CSV files do not match.")

# Calculate the average of the first 720 predictions
df_average_first_720 = pd.DataFrame({
    'id': df1['id'].iloc[:720],
    'prediction': (df1['prediction'].iloc[:720])
})

# Keep the middle predictions from df1 as is
df_average_last = pd.DataFrame({
    'id': df1['id'].iloc[720:],
    'prediction': (df1['prediction'].iloc[720:]+df2['prediction'].iloc[720:])/2
})


# Concatenate the three DataFrames
df_combined = pd.concat([df_average_first_720, df_average_last], ignore_index=True)

# set values smaller than 0.5 to 0 
df_combined.loc[df_combined['prediction'] < 0.5, 'prediction'] = 0

# Save the combined predictions to a new CSV file
df_combined.to_csv('combined_gluon_cat.csv', index=False)
