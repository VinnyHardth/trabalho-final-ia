import pandas as pd

file_path = 'data/trains-data_coded.csv'

df = pd.read_csv(file_path, delimiter=';')

distinct_values = df['Number_of_cars'].unique()

list_df = []
for value in distinct_values:
    filtered_df = df[df['Number_of_cars'] == value]
    list_df.append(filtered_df)

print(list_df)