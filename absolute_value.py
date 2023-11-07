import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('kaggle_submission_catboost_5.csv')

# Ensure there are no negative predictions
df['prediction'] = df['prediction'].apply(lambda x: max(x, 0))

# Set predictions to zero where up to three non-zero predictions are surrounded by zeros
for i in range(1, len(df) - 1):
    # Check single non-zero prediction surrounded by zeros
    if df.loc[i - 1, 'prediction'] == 0 and df.loc[i + 1, 'prediction'] == 0:
        df.loc[i, 'prediction'] = 0
    # Check two consecutive non-zero predictions surrounded by zeros
    if i < len(df) - 2 and df.loc[i - 1, 'prediction'] == 0 and df.loc[i + 2, 'prediction'] == 0:
        df.loc[i, 'prediction'] = 0
        df.loc[i + 1, 'prediction'] = 0
    # Check three consecutive non-zero predictions surrounded by zeros

# Save the updated DataFrame back to a CSV file
df.to_csv('predictions_updated_5.csv', index=False)

print("Single, double, or triple non-zero predictions surrounded by zeros have been set to zero.")
