import pandas as pd

# reading data
file_path = 'openpowerlifting.csv'
data = pd.read_csv(file_path)

# choosing columns
columns_of_interest = ['Name', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg']
data = data[columns_of_interest]

# removing null rows
data = data.dropna(subset=columns_of_interest)

# aggregating data to record only best lifts
aggregated_data = data.groupby('Name', as_index=False).max()

# new csv
aggregated_data.to_csv('processed_powerlifting_data.csv', index=False)