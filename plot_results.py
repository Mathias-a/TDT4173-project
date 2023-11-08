import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv('predictions_updated_7.csv')
df2 = pd.read_csv('predictions_updated_8.csv')
df3 = pd.read_csv('predictions_updated_150.csv')
df4 = pd.read_csv('predictions_updated_155.csv')


# Verify that the IDs match in all three DataFrames (assuming they are sorted and have the same length)
if not (df1['id'].equals(df2['id']) and df1['id'].equals(df3['id'])):
    raise ValueError("The IDs in the CSV files do not match.")

# Create a new DataFrame to hold all predictions
df_predictions = pd.DataFrame({
    'id': df1['id'],
    'Prediction 7': df1['prediction'],
    'Prediction 8': df2['prediction'],
    'Prediction 150': df3['prediction'],
    'Prediction 155': df4['prediction']

})

# Plot the predictions
plt.figure(figsize=(10, 6))

plt.plot(df_predictions['id'], df_predictions['Prediction 7'], label='Prediction 1', alpha=0.8)
plt.plot(df_predictions['id'], df_predictions['Prediction 8'], label='Prediction 2', alpha=0.8)
plt.plot(df_predictions['id'], df_predictions['Prediction 150'], label='Prediction 3', alpha=0.8)
plt.plot(df_predictions['id'], df_predictions['Prediction 155'], label='Prediction 3', alpha=0.8)


plt.title('Comparison of Predictions')
plt.xlabel('ID')
plt.ylabel('Prediction Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
