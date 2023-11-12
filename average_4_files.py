import pandas as pd

# Load the datasets
df1 = pd.read_csv('predictions/autogluon-Erik.csv')
df2 = pd.read_csv('predictions/autogluon-2-stack-params.csv')
df3 = pd.read_csv('predictions/predictions_updated_10.csv')
# Verify that the IDs match in all DataFrames (assuming they are sorted and have the same length)
# if not (df1['id'].equals(df2['id']) and df1['id'].equals(df3['id']) and df1['id']):
#     raise ValueError("The IDs in the CSV files do not match.")

# Calculate the average of the predictions for all rows
df_average = pd.DataFrame({
    'id': df1['id'],
    'prediction': (df1['prediction'] + df3['prediction'] + df2['prediction'])/3
})

# Save the averaged predictions to a new CSV file
df_average.to_csv('predictions/averaged_autogluon_erik_145_mathias.csv', index=False)