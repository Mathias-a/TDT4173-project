import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load the data from the files
file1 = pd.read_csv('predictions_updated_10.csv')
file2 = pd.read_csv('autogluon_attempt_3.csv')

# Ensure the predictions are in the same order by merging on the id column
merged = file1.merge(file2, on='id', suffixes=('_file1', '_file2'))

# Compute the Mean Absolute Error
mae = mean_absolute_error(merged['prediction_file1'][720:], merged['prediction_file2'][720:])
print('Mean Absolute Error:', mae)