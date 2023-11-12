import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv('kaggle_submission_catboost_10.csv')
# df2 = pd.read_csv('kaggle_submission_catboost_11.csv')
df3 = pd.read_csv('145-recreated.csv')
# df4 = pd.read_csv('combined_gluon_cat.csv')
# df5 = pd.read_csv('kaggle_submission_catboost_19.csv')



# Verify that the IDs match in all three DataFrames (assuming they are sorted and have the same length)
# if not (df1['id'].equals(df2['id']) and df1['id'].equals(df3['id'])):
#     raise ValueError("The IDs in the CSV files do not match.")

# Create a new DataFrame to hold all predictions
df_predictions = pd.DataFrame({
    'id': df1['id'],
    'Prediction 145': df1['prediction'],
    'Prediction automl': df3['prediction'],
    # 'combined': df4['prediction'],
    # 'Prediction hopeful': df5['prediction'],
    # 'Prediction long train': df3['prediction']


})

# Plot the predictions
plt.figure(figsize=(10, 6))

plt.plot(df_predictions['id'], df_predictions['Prediction 145'], label='Prediction 145', alpha=0.8)
plt.plot(df_predictions['id'], df_predictions['Prediction automl'], label='ML', alpha=0.8)
# plt.plot(df_predictions['id'], df_predictions['combined'], label='Prediction long train', alpha=0.8)
# plt.plot(df_predictions['id'], df_predictions['Prediction hopeful'], label='Prediction hopeful', alpha=0.8)
# plt.plot(df_predictions['id'], df_predictions['Prediction ensemble'], label='Prediction ensemble', alpha=0.8)



plt.title('Comparison of Predictions')
plt.xlabel('ID')
plt.ylabel('Prediction Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
