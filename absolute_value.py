import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('kaggle_submission_catboost.csv')

# Loop through the DataFrame and set negative values to zero
df['prediction'] = df['prediction'].apply(lambda x: max(x, 0))

# Save the updated DataFrame back to a CSV file
df.to_csv('predictions_updated.csv', index=False)

print("Negative values in 'prediction' column have been set to zero.")