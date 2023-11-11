import pandas as pd

# Load the CSV file into a DataFrame
# df = pd.read_csv('combined_predictions_A_ensembled_BC_specialized.csv')
df = pd.read_csv('combined_gluon_cat.csv')

# Ensure there are no negative predictions
df['prediction'] = df['prediction'].apply(lambda x: max(x, 0))

df['prediction'] = df['prediction'].apply(lambda x: 0 if x < 0.1 else x)


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
df.to_csv('updated_combined_gluon_cat.csv', index=False)
# df.to_csv('combined_predictions_A_ensembled_BC_specialized.csv', index=False)

print("Single, double, or triple non-zero predictions surrounded by zeros have been set to zero.")
